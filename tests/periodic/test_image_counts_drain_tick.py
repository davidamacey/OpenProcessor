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
from src.services.periodic import image_counts_drain, runner


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


def test_a_bound_project_pass_recounts_the_marked_and_the_uncounted(
    image_counts_dirty: dict[str, set[str]],
) -> None:
    fake = _world()
    image_counts_dirty['op:image_counts:dirty:default'] = {'img-a'}
    asyncio.run(image_counts_drain.drain_bound_project(fake, FakeValkeySets(image_counts_dirty)))
    docs = fake.docs(IMAGES)
    assert docs['img-a']['counts']['items'] == 1
    assert docs['img-b']['counts']['items'] == 0
    assert image_counts_dirty == {}


def test_a_tick_builds_one_client_of_each_kind_however_many_projects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clients built per tick must not grow with the project count (it was one pair per project)."""
    opened: list[str] = []
    fakes = _Closing({})

    async def opensearch(_timeout: float | None = None) -> Any:
        opened.append('opensearch')
        return fakes

    def valkey() -> FakeValkeySets:
        opened.append('valkey')
        return FakeValkeySets()

    async def each(_lock: int, body: Any, *, task: str) -> None:
        for slug in ('a', 'b', 'c', 'd', 'e'):
            await body(SimpleNamespace(slug=slug))

    async def drain_one(opensearch: Any, valkey: Any, budget_s: float) -> None:
        assert opensearch is fakes

    monkeypatch.setattr(runner, 'opensearch_client', opensearch)
    monkeypatch.setattr(image_counts_drain, 'valkey_client', valkey)
    monkeypatch.setattr(image_counts_drain, 'drain_bound_project', drain_one)
    monkeypatch.setattr(image_counts_drain, 'for_each_project', each)
    monkeypatch.setattr(image_counts_drain, 'read_tick_projects', lambda: [1] * 5)
    asyncio.run(image_counts_drain.tick())
    assert opened == ['valkey', 'opensearch']
    assert fakes.closed


def test_a_tick_with_no_project_builds_no_opensearch_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[int] = []

    async def opensearch(_timeout: float | None = None) -> Any:
        built.append(1)
        return _Closing({})

    async def each(_lock: int, body: Any, *, task: str) -> None:
        return None

    monkeypatch.setattr(runner, 'opensearch_client', opensearch)
    monkeypatch.setattr(image_counts_drain, 'valkey_client', FakeValkeySets)
    monkeypatch.setattr(image_counts_drain, 'for_each_project', each)
    monkeypatch.setattr(image_counts_drain, 'read_tick_projects', list)
    asyncio.run(image_counts_drain.tick())
    assert built == []


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

    async def drain_one(_os: Any, _vk: Any, budget_s: float | None = None) -> None:
        seen.append(budget_s)
        # The first project uses its whole share and a little more (a batch in flight).
        clock.now += (budget_s or 0) + 1 if len(seen) == 1 else 0

    async def each(_lock: int, body: Any, *, task: str) -> None:
        for project in projects:
            await body(project)

    monkeypatch.setattr(image_counts_drain, 'drain_bound_project', drain_one)
    monkeypatch.setattr(image_counts_drain, 'for_each_project', each)
    monkeypatch.setattr(image_counts_drain, 'read_tick_projects', lambda: projects)
    monkeypatch.setattr(image_counts_drain, 'valkey_client', FakeValkeySets)
    monkeypatch.setattr(image_counts_drain.time, 'monotonic', clock)
    asyncio.run(image_counts_drain.tick())
    total = image_counts_drain.TICK_BUDGET_S
    assert seen[0] is not None
    assert seen[0] == total / 3
    # The overrun of the first project comes out of what is left (split over the two to go);
    # a project that finishes at once leaves its share to the one after it.
    assert seen[1] == (total - seen[0] - 1) / 2
    assert seen[2] == total - seen[0] - 1
