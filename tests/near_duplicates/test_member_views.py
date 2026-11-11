"""The group view carries what a page draws: pictures, class, holdout state, the pin to the
representative, the decisions list, and why an empty answer is empty."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.db.session import async_session_factory
from src.schemas.curation import near_duplicates as wire
from src.services.near_duplicates import member_views, status as service
from src.services.near_duplicates.build import full_build

from .engine import IMAGES, image_docs, make_env


@pytest.fixture
def api_base(monkeypatch: pytest.MonkeyPatch) -> str:
    base = '/curation/projects/nd'
    monkeypatch.setattr(member_views, 'project_api_base', lambda: base)
    return base


def test_a_thumbnail_is_the_r3_image_route_at_the_image_level_and_the_crop_route_at_the_item_level(
    api_base: str,
) -> None:
    assert member_views.thumbnail_url('image', 'img-1') == f'{api_base}/images/img-1/thumbnail'
    assert member_views.thumbnail_url('item', 'crop-1') == f'{api_base}/crops/crop-1/thumbnail'


@pytest.mark.asyncio
async def test_item_facts_read_class_image_and_holdout_in_one_query() -> None:
    os_client = AsyncMock()
    os_client.search = AsyncMock(
        return_value={
            'hits': {
                'hits': [
                    {'_id': 'c1', '_source': {'image_id': 'i1', 'class_name': 'van'}},
                    {
                        '_id': 'c2',
                        '_source': {'image_id': 'i2', 'class_name': None, 'test_holdout': True},
                    },
                ]
            }
        }
    )
    facts = await member_views.item_facts(os_client, 'item', ['c1', 'c2'])
    assert os_client.search.await_count == 1
    assert facts['c1'] == {'image_id': 'i1', 'class_name': 'van', 'in_holdout': False}
    assert facts['c2']['in_holdout'] is True


@pytest.mark.asyncio
async def test_an_image_is_in_the_holdout_when_an_item_standing_on_it_is() -> None:
    os_client = AsyncMock()
    os_client.search = AsyncMock(
        return_value={'aggregations': {'held': {'buckets': [{'key': 'i2', 'doc_count': 1}]}}}
    )
    facts = await member_views.item_facts(os_client, 'image', ['i1', 'i2'])
    assert os_client.search.await_count == 1
    assert (facts['i1']['in_holdout'], facts['i2']['in_holdout']) == (False, True)
    assert facts['i2']['image_id'] == 'i2'
    assert facts['i2']['class_name'] is None


def test_a_member_view_of_an_item_names_its_crop_and_its_pin(api_base: str) -> None:
    pin = wire.NearDupPin(id='d1', kind='different')
    view = member_views.member_view(
        'item',
        ('c1', False, 0.99),
        {'c1': {'image_id': 'i1', 'class_name': 'van', 'in_holdout': True}},
        {'c1': pin},
    )
    assert (view.crop_id, view.image_id, view.class_name, view.in_holdout) == (
        'c1',
        'i1',
        'van',
        True,
    )
    assert view.decision == pin
    assert view.thumbnail_url == f'{api_base}/crops/c1/thumbnail'


@pytest.fixture(autouse=True)
def _space(monkeypatch: pytest.MonkeyPatch) -> None:
    from .engine import plan

    async def resolved(_target, _key=None):
        return SimpleNamespace(space_key=plan().space_key)

    monkeypatch.setattr(service, 'embedding_space', resolved)


@pytest.mark.db
@pytest.mark.asyncio
async def test_the_decisions_list_is_newest_first_and_pages_by_cursor(project) -> None:
    for a, b in [('a', 'b'), ('c', 'd'), ('e', 'f')]:
        async with async_session_factory()() as session:
            await service.decide(
                session,
                project,
                wire.NearDupDecisionIn(level='image', a_id=a, b_id=b, decision='same'),
                'tester',
            )
    async with async_session_factory()() as session:
        first = await service.decisions(
            session, project, level='image', member_id=None, cursor=None, limit=2
        )
    assert [(d.a_id, d.b_id) for d in first.items] == [('e', 'f'), ('c', 'd')]
    assert first.next_cursor is not None
    async with async_session_factory()() as session:
        second = await service.decisions(
            session, project, level='image', member_id=None, cursor=first.next_cursor, limit=2
        )
    assert [(d.a_id, d.b_id) for d in second.items] == [('a', 'b')]
    assert second.next_cursor is None
    async with async_session_factory()() as session:
        named = await service.decisions(
            session, project, level=None, member_id='d', cursor=None, limit=10
        )
    assert [(d.a_id, d.b_id) for d in named.items] == [('c', 'd')]


@pytest.mark.db
@pytest.mark.asyncio
async def test_decisions_of_another_project_are_not_listed(project) -> None:
    from .conftest import make_project

    async with async_session_factory()() as session:
        await service.decide(
            session,
            project,
            wire.NearDupDecisionIn(level='image', a_id='a', b_id='b', decision='same'),
            'tester',
        )
    other = await make_project()
    async with async_session_factory()() as session:
        page = await service.decisions(
            session, other, level=None, member_id=None, cursor=None, limit=10
        )
    assert page.items == []


@pytest.mark.db
@pytest.mark.asyncio
async def test_a_group_detail_serves_the_pin_to_the_representative(
    pg, project, api_base: str
) -> None:
    docs = {IMAGES: image_docs({'g': ['a1', 'a2', 'a3']}, ['s1'])}
    env, _ = await make_env(pg, project, docs)
    await full_build(env)
    os_client = AsyncMock(
        search=AsyncMock(
            return_value={'aggregations': {'held': {'buckets': [{'key': 'a2', 'doc_count': 1}]}}}
        )
    )
    async with async_session_factory()() as session:
        page = await service.groups(
            session, project, level='image', purpose='dedup', min_size=2, has_holdout=None,
            cursor=None, limit=10,
        )  # fmt: skip
    group = page.items[0]
    assert group.representative_thumbnail_url == (
        f'{api_base}/images/{group.representative_id}/thumbnail'
    )
    gid = uuid.UUID(group.group_id)
    async with async_session_factory()() as session:
        before = await service.group(session, project, os_client, 'image', gid)
    assert all(m.decision is None for m in before.members)
    other = next(m.member_id for m in before.members if not m.is_representative)
    async with async_session_factory()() as session:
        pinned = await service.decide(
            session,
            project,
            wire.NearDupDecisionIn(
                level='image', a_id=group.representative_id, b_id=other, decision='different'
            ),
            'tester',
        )
    async with async_session_factory()() as session:
        after = await service.group(session, project, os_client, 'image', gid)
    by_id = {m.member_id: m for m in after.members}
    assert by_id[other].decision == wire.NearDupPin(id=pinned.id, kind='different')
    assert by_id[group.representative_id].decision is None
    assert by_id['a2'].in_holdout is True
    assert by_id['a1'].in_holdout is False
    assert all(m.image_id == m.member_id and m.crop_id is None for m in after.members)


@pytest.mark.db
@pytest.mark.asyncio
async def test_an_empty_page_names_the_cut_and_the_scored_count(pg, project) -> None:
    docs = {IMAGES: image_docs({}, ['s1', 's2', 's3'])}
    env, _ = await make_env(pg, project, docs)
    await full_build(env)
    async with async_session_factory()() as session:
        page = await service.groups(
            session, project, level='image', purpose='dedup', min_size=2, has_holdout=None,
            cursor=None, limit=10,
        )  # fmt: skip
    assert page.items == []
    assert page.empty_reason == '0 dedup groups at cosine 0.98 among 3 images'


@pytest.mark.db
@pytest.mark.asyncio
async def test_a_level_with_no_run_says_to_build_it(project) -> None:
    async with async_session_factory()() as session:
        page = await service.groups(
            session, project, level='item', purpose='dedup', min_size=2, has_holdout=None,
            cursor=None, limit=10,
        )  # fmt: skip
    assert page.empty_reason == 'no item-level run yet: build it to find near-duplicates'
