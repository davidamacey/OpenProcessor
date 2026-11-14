"""A consumer that stops reading must not hang the pager, even with a full page queue."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from typing import Any, cast

import pytest

from src.services.curation.clustering.pit_pages import iter_pages
from tests.curation.vector_os_fake import VectorOS, random_docs


@pytest.mark.asyncio
async def test_stopping_after_the_first_page_closes_the_pager_and_deletes_the_point_in_time() -> (
    None
):
    client = VectorOS(random_docs(12, seed=9))

    async def read_one_page() -> None:
        pager = iter_pages(
            cast('Any', client), index='i', query={}, source=['image_id'], n_slices=1, page_size=1
        )
        async with aclosing(pager) as pages:
            async for _page in pages:
                break

    await asyncio.wait_for(read_one_page(), timeout=5)
    assert client.pits_deleted == 1
    others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert others == []
