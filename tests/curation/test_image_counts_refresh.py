"""Stored per-image counts stay equal to the items (R-13): marking, the drain, idempotence.

An item write marks its image in a Valkey set; a drain recounts the marked images with the same
aggregation the gallery cards use and stores the result on the image documents.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fixtures.fake_valkey import FakeValkeySets

from curation.query_fakes import QueryFakeOpenSearch
from src.clients.image_counts_dirty import DirtyImages, dirty_key
from src.clients.occ import occ_skip_on_conflict_bulk, occ_update_one
from src.clients.occ_bulk import occ_update_bulk
from src.clients.occ_upsert import occ_upsert_bulk
from src.config import IndexRole, index_name
from src.config.curation_accessors import base_curation_config
from src.services.curation import image_counts_refresh as refresh


CFG = base_curation_config()
ITEMS = CFG.items_index
IMAGES = index_name(CFG, IndexRole.IMAGES)
DIRTY = 'op:image_counts:dirty:default'


def _item(n: str, image: str, **extra: Any) -> dict[str, Any]:
    return {'crop_id': n, 'image_id': image, 'bbox_norm': [0.1, 0.1, 0.5, 0.5], **extra}


def _world() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {
            ITEMS: {
                'a1': _item('a1', 'img-a', class_id=3, class_name='sedan', class_validated=True),
                'a2': _item('a2', 'img-a'),
                'a3': _item('a3', 'img-a', class_id=3, class_excluded=True),
                'b1': _item('b1', 'img-b', class_id=4),
                'b2': _item('b2', 'img-b', class_id=4, test_holdout=True),
                'c1': _item('c1', 'img-c'),
            },
            IMAGES: {
                n: {'image_id': n, 'image_path': f'/d/{n}.jpg'}
                for n in ('img-a', 'img-b', 'img-c', 'img-d')
            },
        }
    )


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _drain(fake: QueryFakeOpenSearch, dirty_data: dict[str, set[str]]) -> refresh.DrainReport:
    return _run(refresh.drain(fake, DirtyImages(FakeValkeySets(dirty_data))))


def test_label_write_marks_the_image_dirty(image_counts_dirty: dict[str, set[str]]) -> None:
    fake = _world()
    _run(
        occ_update_one(
            fake,
            doc_id='a2',
            merger=lambda _s: {'class_id': 3, 'class_validated': True},
            vector_excludes=[],
        )
    )
    assert image_counts_dirty == {DIRTY: {'img-a'}}

    _run(
        occ_skip_on_conflict_bulk(
            fake, doc_ids=['b1', 'c1'], merger=lambda _i, _s: {'class_id': 5}, vector_excludes=[]
        )
    )
    assert image_counts_dirty == {DIRTY: {'img-a', 'img-b', 'img-c'}}


def test_a_human_bulk_label_marks_each_image_once(image_counts_dirty: dict[str, set[str]]) -> None:
    fake = _world()
    status = _run(
        occ_update_bulk(
            fake,
            ids=['a1', 'a2', 'a3'],
            merge_fn=lambda _i, _s: {'class_id': 9},
            vector_excludes=[],
        )
    )
    assert set(status.values()) == {'updated'}
    assert image_counts_dirty == {DIRTY: {'img-a'}}


def test_a_write_that_cannot_change_a_count_marks_nothing(
    image_counts_dirty: dict[str, set[str]],
) -> None:
    fake = _world()
    _run(
        occ_update_one(
            fake,
            doc_id='a2',
            merger=lambda _s: {'updated_at': 'now', 'blur_lap_ratio': 0.5},
            vector_excludes=[],
        )
    )
    assert image_counts_dirty == {}


def test_an_ingested_item_marks_its_image(image_counts_dirty: dict[str, set[str]]) -> None:
    fake = _world()
    _run(
        occ_upsert_bulk(
            fake,
            [_item('d1', 'img-d')],
            human_field_guards=[],
            vector_excludes=[],
            vector_field='embedding',
            state_field='embedding_state',
        )
    )
    assert image_counts_dirty == {DIRTY: {'img-d'}}


def test_drain_writes_counts_equal_to_an_items_aggregate(
    image_counts_dirty: dict[str, set[str]],
) -> None:
    fake = _world()
    image_counts_dirty[DIRTY] = {'img-a', 'img-b', 'img-d'}
    report = _drain(fake, image_counts_dirty)
    docs = fake.docs(IMAGES)
    assert docs['img-a']['counts'] == {
        'items': 2,
        'labeled': 1,
        'validated': 1,
        'unlabeled': 1,
        'in_holdout': 0,
    }
    # img-b: the held-out item is not among `items`, but is counted as held out.
    assert docs['img-b']['counts'] == {
        'items': 1,
        'labeled': 1,
        'validated': 0,
        'unlabeled': 0,
        'in_holdout': 1,
    }
    assert docs['img-d']['counts'] == dict.fromkeys(refresh_keys(), 0)
    assert all(docs[i]['counts_at'] for i in ('img-a', 'img-b', 'img-d'))
    assert report.recounted == 3
    assert image_counts_dirty == {}


def refresh_keys() -> tuple[str, ...]:
    from src.clients.curation_opensearch.count_fields import COUNT_KEYS

    return COUNT_KEYS


def test_the_drain_refreshes_the_items_index_before_it_counts(
    image_counts_dirty: dict[str, set[str]],
) -> None:
    fake = _world()
    image_counts_dirty[DIRTY] = {'img-a'}
    _drain(fake, image_counts_dirty)
    assert fake.indices.refreshed == [ITEMS]


def test_drain_is_idempotent(image_counts_dirty: dict[str, set[str]]) -> None:
    fake = _world()
    image_counts_dirty[DIRTY] = {'img-a', 'img-b'}
    _drain(fake, image_counts_dirty)
    first = {i: dict(d['counts']) for i, d in fake.docs(IMAGES).items() if 'counts' in d}
    # The same marks again (a retried write, a drain that died before removing them).
    image_counts_dirty[DIRTY] = {'img-a', 'img-b'}
    _drain(fake, image_counts_dirty)
    again = {i: dict(d['counts']) for i, d in fake.docs(IMAGES).items() if 'counts' in d}
    assert again == first


def test_a_mark_made_after_the_claim_is_not_lost(image_counts_dirty: dict[str, set[str]]) -> None:
    fake = _world()
    image_counts_dirty[DIRTY] = {'img-a'}
    dirty = DirtyImages(FakeValkeySets(image_counts_dirty))

    async def run() -> None:
        assert await dirty.claim()
        # A label lands while the drain works on the claimed set.
        await FakeValkeySets(image_counts_dirty).sadd(dirty_key(), 'img-b')
        await refresh.recount_images(fake, await dirty.batch(10))
        await dirty.done(['img-a'])

    _run(run())
    assert image_counts_dirty == {DIRTY: {'img-b'}}


def test_a_claimed_set_left_by_a_dead_drain_is_worked_first(
    image_counts_dirty: dict[str, set[str]],
) -> None:
    fake = _world()
    image_counts_dirty['op:image_counts:claimed:default'] = {'img-a'}
    image_counts_dirty[DIRTY] = {'img-b'}
    _drain(fake, image_counts_dirty)
    assert 'counts' in fake.docs(IMAGES)['img-a']
    assert image_counts_dirty == {DIRTY: {'img-b'}}


def test_the_sweep_counts_images_that_were_never_counted(
    image_counts_dirty: dict[str, set[str]],
) -> None:
    fake = _world()
    fake.docs(IMAGES)['img-c']['counts_at'] = '2026-01-01T00:00:00'
    fake.docs(IMAGES)['img-c']['counts'] = {'items': 99}
    report = _drain(fake, image_counts_dirty)
    docs = fake.docs(IMAGES)
    assert report.swept == 3
    assert docs['img-c']['counts'] == {'items': 99}
    assert docs['img-a']['counts']['items'] == 2
    assert _drain(fake, image_counts_dirty).swept == 0


def _refusing(fake: QueryFakeOpenSearch, monkeypatch: Any, refused: set[str]) -> None:
    """Make the images index refuse the count write of every id in ``refused`` (HTTP 400)."""
    real = fake.bulk

    async def bulk(*, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        ids = [action['update']['_id'] for action in body[::2]]
        ok = [i for i in ids if i not in refused]
        good = await real(body=[part for i in ok for part in body[2 * ids.index(i) :][:2]], **kw)
        by_id = dict(zip(ok, good['items'], strict=True))
        refusal = {'update': {'status': 400, 'error': {'type': 'x'}}}
        items = [by_id.get(i, refusal) for i in ids]
        return {'errors': True, 'items': items}

    monkeypatch.setattr(fake, 'bulk', bulk)


def test_a_refused_count_write_requeues_its_image_and_does_not_wedge_the_drain(
    image_counts_dirty: dict[str, set[str]], monkeypatch: Any
) -> None:
    fake = _world()
    _refusing(fake, monkeypatch, {'img-a'})
    image_counts_dirty[DIRTY] = {'img-a', 'img-b'}
    report = _drain(fake, image_counts_dirty)
    docs = fake.docs(IMAGES)
    # The accepted image was counted and left the set; the refused one went back to the dirty
    # set, so no claimed set is left for the next tick to resume forever.
    assert 'counts' in docs['img-b']
    assert 'counts' not in docs['img-a']
    assert image_counts_dirty == {DIRTY: {'img-a'}}
    assert (report.recounted, report.refused) == (1, 1)
    # The sweep still ran in the same pass (it counts the never-counted images).
    assert 'counts' in docs['img-c']
    assert report.swept == 2  # img-c and img-d; img-a is refused there too

    # A write lands while the refusal persists: the next pass claims it, so nothing piles up.
    image_counts_dirty[DIRTY] |= {'img-c'}
    second = _drain(fake, image_counts_dirty)
    assert second.recounted == 1
    assert image_counts_dirty == {DIRTY: {'img-a'}}


def test_a_refused_image_is_counted_once_the_index_accepts_it_again(
    image_counts_dirty: dict[str, set[str]], monkeypatch: Any
) -> None:
    fake = _world()
    real_bulk = fake.bulk
    _refusing(fake, monkeypatch, {'img-a'})
    image_counts_dirty[DIRTY] = {'img-a'}
    _drain(fake, image_counts_dirty)
    monkeypatch.setattr(fake, 'bulk', real_bulk)
    _drain(fake, image_counts_dirty)
    assert fake.docs(IMAGES)['img-a']['counts']['items'] == 2
    assert image_counts_dirty == {}


def test_the_sweep_size_is_a_setting_and_zero_turns_the_sweep_off(
    image_counts_dirty: dict[str, set[str]], monkeypatch: Any
) -> None:
    fake = _world()
    monkeypatch.setenv('OP_IMAGE_COUNTS_SWEEP_PER_TICK', '2')
    assert _drain(fake, image_counts_dirty).swept == 2
    monkeypatch.setenv('OP_IMAGE_COUNTS_SWEEP_PER_TICK', '0')
    assert _drain(fake, image_counts_dirty).swept == 0
    assert sum('counts_at' in d for d in fake.docs(IMAGES).values()) == 2


def test_projects_never_share_a_dirty_set(image_counts_dirty: dict[str, set[str]]) -> None:
    from src.clients.image_counts_dirty import mark_dirty
    from src.config.project_context import bind_project

    class _Record:
        slug = 'other'

    _run(mark_dirty(['img-x']))
    with bind_project(_Record()):  # type: ignore[arg-type]
        _run(mark_dirty(['img-y']))
        dirty = DirtyImages(FakeValkeySets(image_counts_dirty))
        assert _run(dirty.claim()) is True
        assert _run(dirty.batch(10)) == ['img-y']
    assert image_counts_dirty['op:image_counts:dirty:default'] == {'img-x'}


def test_a_valkey_outage_does_not_fail_the_write_that_marks(monkeypatch: Any) -> None:
    import redis

    from src.clients import image_counts_dirty

    class _Down:
        closed = False

        async def sadd(self, *_a: Any) -> int:
            raise redis.ConnectionError('valkey is down')

        async def aclose(self) -> None:
            self.closed = True

    client = _Down()
    monkeypatch.setattr(image_counts_dirty, 'valkey_client', lambda: client)
    assert _run(image_counts_dirty.mark_dirty(['img-a'])) == 0
    assert client.closed
