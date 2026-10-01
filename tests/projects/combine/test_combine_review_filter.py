"""``combine_conflict=true`` on the review tabs lists exactly the boxes a
combine flagged (owner decision D6)."""

from __future__ import annotations

import pytest

from src.services.curation import review_queries
from src.services.curation.review_request import ReviewFilters, build_review_request

from .test_combine_execute import MAPPING, build
from .world import World, run_job


@pytest.mark.asyncio
async def test_the_filter_lists_only_flagged_items(world: World) -> None:
    build(world)
    await run_job(world, world.request(['cars-a', 'cars-b'], MAPPING))
    index = world.items_index('combined')
    # An item the queue shows for another reason, so the filter has work to do.
    unvalidated = (
        d
        for d in world.items('combined').values()
        if not d.get('combine_conflict') and not d['class_validated']
    )
    other = dict(next(unvalidated))
    other.update(crop_id='outlier', cluster_distance=0.9)
    world.items('combined')['outlier'] = other

    async def queue(filters: ReviewFilters) -> list[dict]:
        req = await build_review_request('all', filters, None, world.fake)
        resp = await world.fake.search(index=index, body={'size': 50, 'query': req.query})
        return [h['_source'] for h in resp['hits']['hits']]

    everything = await queue(ReviewFilters(include_test=True))
    flagged = await queue(ReviewFilters(include_test=True, combine_conflict=True))
    assert len(flagged) == 1
    assert flagged[0]['combine_conflict'] is True
    assert len(everything) == 2  # the conflict, plus the outlier
    assert 'combine_conflict' in review_queries.COMMON_FILTERS
