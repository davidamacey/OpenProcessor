"""``op.periodic.image_counts_drain`` for one project: claim its dirty set, recount, sweep.

The per-project wrapper (advisory lock, one pass per active project) is the shared
``for_each_project`` frame, covered by the periodic db tier.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
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


class _Clock:
    now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_the_tick_budget_is_shared_so_every_project_gets_a_fair_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One drain budget per project summed past the task's soft limit (a busy project starved the
    later ones until the tick was killed); the tick's budget is split over its projects."""
    projects = [SimpleNamespace(slug=s) for s in ('a', 'b', 'c')]
    clock = _Clock()
    seen: list[float | None] = []

    async def drain_one(budget_s: float | None = None) -> None:
        seen.append(budget_s)
        # The first project uses its whole share and a little more (a batch in flight).
        clock.now += (budget_s or 0) + 1 if len(seen) == 1 else 0

    async def each(_lock: int, body: Any, *, task: str) -> None:
        for project in projects:
            await body(project)

    monkeypatch.setattr(image_counts_drain, 'drain_bound_project', drain_one)
    monkeypatch.setattr(image_counts_drain, 'for_each_project', each)
    monkeypatch.setattr(image_counts_drain, 'read_tick_projects', lambda: projects)
    monkeypatch.setattr(image_counts_drain.time, 'monotonic', clock)
    asyncio.run(image_counts_drain.tick())
    total = image_counts_drain.TICK_BUDGET_S
    assert seen[0] is not None
    assert seen[0] == total / 3
    # The overrun of the first project comes out of what is left (split over the two to go);
    # a project that finishes at once leaves its share to the one after it.
    assert seen[1] == (total - seen[0] - 1) / 2
    assert seen[2] == total - seen[0] - 1
