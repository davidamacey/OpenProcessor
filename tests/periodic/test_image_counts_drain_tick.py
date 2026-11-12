"""``op.periodic.image_counts_drain`` for one project: claim its dirty set, recount, sweep.

The per-project wrapper (advisory lock, one pass per active project) is the shared
``for_each_project`` frame, covered by the periodic db tier.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from curation.query_fakes import QueryFakeOpenSearch
from fixtures.fake_valkey import FakeValkeySets

from src.config import IndexRole, index_name
from src.config.curation_accessors import base_curation_config
from src.services.periodic import image_counts_drain


if TYPE_CHECKING:
    import pytest


CFG = base_curation_config()
ITEMS = CFG.items_index
IMAGES = index_name(CFG, IndexRole.IMAGES)


class _Closing(QueryFakeOpenSearch):
    closed = False

    async def close(self) -> None:
        self.closed = True


def _world() -> _Closing:
    return _Closing(
        {
            ITEMS: {'a1': {'crop_id': 'a1', 'image_id': 'img-a', 'class_id': 1}},
            IMAGES: {'img-a': {'image_id': 'img-a'}, 'img-b': {'image_id': 'img-b'}},
        }
    )


def test_a_bound_project_pass_recounts_the_marked_and_the_uncounted_and_closes_its_client(
    monkeypatch: pytest.MonkeyPatch, image_counts_dirty: dict[str, set[str]]
) -> None:
    fake = _world()
    image_counts_dirty['op:image_counts:dirty:default'] = {'img-a'}

    async def client() -> Any:
        return fake

    monkeypatch.setattr(image_counts_drain, 'opensearch_client', client)
    monkeypatch.setattr(
        image_counts_drain, 'valkey_client', lambda: FakeValkeySets(image_counts_dirty)
    )
    asyncio.run(image_counts_drain.drain_bound_project())
    docs = fake.docs(IMAGES)
    assert docs['img-a']['counts']['items'] == 1
    assert docs['img-b']['counts']['items'] == 0
    assert image_counts_dirty == {}
    assert fake.closed


def test_the_drain_opens_its_valkey_client_once_per_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[FakeValkeySets] = []

    def factory() -> FakeValkeySets:
        seen.append(FakeValkeySets())
        return seen[-1]

    async def client() -> Any:
        return _world()

    monkeypatch.setattr(image_counts_drain, 'valkey_client', factory)
    monkeypatch.setattr(image_counts_drain, 'opensearch_client', client)
    asyncio.run(image_counts_drain.drain_bound_project())
    assert len(seen) == 1
