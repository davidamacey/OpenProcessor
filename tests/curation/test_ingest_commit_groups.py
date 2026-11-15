"""The commit step of an ingest, in the pieces a commit group is made of (plan 13a, lane L4).

``commit_group`` is the group transaction (placement read, content claims, pending rows, one
session); ``write_group`` and ``write_prepared`` write the photos; ``finish_commit`` releases what
must be released and announces what was stored, in a ``finally`` the caller always reaches.
``index_items`` (a batch of one: a reprocess, a dataset import, the open-vocabulary run) calls them
in that order and behaves as it did.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from sqlalchemy.exc import OperationalError, TimeoutError as PoolTimeoutError

from curation.ingest_fakes import _jpeg_bytes, _make_service
from curation.ingest_session_fakes import Sessions, count_sessions
from src.services.clustering.assign import Placement
from src.services.curation import ingest_index
from src.services.curation.ingest_index import IndexOutcome
from src.services.curation.ingest_models import IngestResult
from src.services.curation.ingest_policy import IngestPolicy
from src.services.curation.ingest_writer import IngestWriteError
from src.services.curation.item_doc import DetectedItem
from src.services.near_duplicates import content_key, ingest_claim
from src.services.near_duplicates.content_key import ClaimOutcome, content_image_id, content_imohash


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> Sessions:
    return count_sessions(monkeypatch)


def _photo(n: int, *, created: bool = True, embedded: bool = True) -> Any:
    """A prepared photo as the group functions read it: identity, items and their ids."""
    data = f'photo-{n}'.encode()
    imohash = content_imohash(data)
    items = [DetectedItem(bbox_pixel=(0, 0, 5, 5), score=0.9) for _ in range(2)]
    if embedded:
        for item in items:
            item.pe_embedding = np.ones(3, dtype=np.float32)
    ctx = SimpleNamespace(
        imohash=imohash,
        image_id=content_image_id(imohash),
        image_path=f'/p/{n}.jpg',
        content=data,
        created=created,
    )
    return SimpleNamespace(ctx=ctx, items=items, cids=[f'{n}-a', f'{n}-b'])


_SERVICE: Any = SimpleNamespace(policy=IngestPolicy())


@pytest.mark.asyncio
async def test_a_group_places_claims_and_marks_pending_in_one_session(sessions: Sessions) -> None:
    group = [_photo(0), _photo(1), _photo(2, created=False)]

    outcomes = await ingest_index.commit_group(_SERVICE, group)

    assert [o.claim for o in outcomes] == [ClaimOutcome()] * 3
    assert sessions.opened == 1
    (session,) = sessions.made
    assert sessions.placed_in == [session]  # one placement read for the whole group
    assert sessions.pending_in == [session, session]  # item level, image level


@pytest.mark.asyncio
async def test_an_empty_group_opens_nothing(sessions: Sessions) -> None:
    assert await ingest_index.commit_group(_SERVICE, []) == []
    assert sessions.opened == 0


@pytest.mark.asyncio
async def test_a_group_whose_transaction_fails_as_a_whole_falls_back_for_that_group_only(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = [_photo(0), _photo(1)], [_photo(2), _photo(3)]
    seen_ids: list[list[str]] = []
    real = ingest_claim.claim_and_mark_in_session

    async def claim(session: Any, project: Any, photos: Any) -> Any:
        ids = [p.item_ids[0] for p in photos]
        seen_ids.append(ids)
        if ids[0] == '2-a':
            raise OperationalError('claim', {}, Exception('server closed the connection'))
        return await real(session, project, photos)

    each: list[list[str]] = []

    async def claim_each(photos: Any) -> Any:
        each.append([p.item_ids[0] for p in photos])
        return [ClaimOutcome() for _ in photos]

    async def place(session: Any, *, item_ids: Any, **_k: Any) -> list[Placement]:
        return [
            Placement(i, uuid.UUID(int=1), uuid.UUID(int=2), 'assigned', 0.9, 0.5) for i in item_ids
        ]

    counted: list[str] = []
    monkeypatch.setattr(ingest_claim, 'claim_and_mark_in_session', claim)
    monkeypatch.setattr(ingest_claim, 'claim_and_mark_each', claim_each)
    monkeypatch.setattr(ingest_index, 'place_new', place)
    monkeypatch.setattr(
        ingest_index, 'count_batch_fallback', lambda stage, exc: counted.append(stage)
    )

    assert len(await ingest_index.commit_group(_SERVICE, first)) == 2
    assert each == []  # the first group's transaction held
    assert len(await ingest_index.commit_group(_SERVICE, second)) == 2

    assert each == [['2-a', '3-a']]  # the failed group runs photo by photo, and only it
    assert counted == ['claim']
    assert all(item.placement is not None for p in second for item in p.items)  # kept on fall-back


@pytest.mark.asyncio
async def test_a_pool_timeout_of_the_group_transaction_reaches_the_caller(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def busy(*_args: Any) -> Any:
        raise PoolTimeoutError('QueuePool limit reached')

    each: list[int] = []

    async def claim_each(photos: Any) -> Any:
        each.append(len(photos))
        return []

    monkeypatch.setattr(ingest_claim, 'claim_and_mark_in_session', busy)
    monkeypatch.setattr(ingest_claim, 'claim_and_mark_each', claim_each)
    with pytest.raises(PoolTimeoutError):
        await ingest_index.commit_group(_SERVICE, [_photo(0)])
    assert each == []


@pytest.mark.asyncio
async def test_a_failed_placement_read_fails_the_group_and_claims_nothing(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An item must never lose its placement unseen: no fall-back to per-photo claims."""

    async def place(*_a: Any, **_k: Any) -> Any:
        raise OperationalError('place', {}, Exception('server closed the connection'))

    claimed: list[Any] = []

    async def claim(*args: Any) -> Any:
        claimed.append(args)
        return []

    async def claim_each(photos: Any) -> Any:
        claimed.append(photos)
        return []

    monkeypatch.setattr(ingest_index, 'place_new', place)
    monkeypatch.setattr(ingest_claim, 'claim_and_mark_in_session', claim)
    monkeypatch.setattr(ingest_claim, 'claim_and_mark_each', claim_each)
    with pytest.raises(OperationalError):
        await ingest_index.commit_group(_SERVICE, [_photo(0)])
    assert claimed == []


def _outcome(
    created: list[str], path: str = '/p/x.jpg', *, status: str = 'success'
) -> IndexOutcome:
    return IndexOutcome(
        result=IngestResult(status=status, image_path=path),
        created_ids=created,
    )


class _Publisher:
    """The service as ``finish_commit`` sees it: it announces through ``_publish_created``."""

    def __init__(self, fail: bool = False) -> None:
        self.announced: list[list[tuple[str, str]]] = []
        self.fail = fail

    async def _publish_created(self, crops: Any) -> None:
        self.announced.append(list(crops))
        if self.fail:
            raise RuntimeError('publish failed')


@pytest.mark.asyncio
async def test_finish_commit_releases_then_announces_the_created_ids_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    async def release(ids: Any) -> int:
        order.append(f'release {sorted(ids)}')
        return 1

    monkeypatch.setattr(content_key, 'release_bound', release)
    service = _Publisher()
    original = service._publish_created

    async def publish(crops: Any) -> None:
        order.append('publish')
        await original(crops)

    service._publish_created = publish  # type: ignore[method-assign]
    written = [
        _outcome(['a', 'b'], '/p/1.jpg'),
        _outcome([], '/p/2.jpg', status='failed'),
        _outcome(['c'], '/p/3.jpg'),
    ]

    await ingest_index.finish_commit(service, written, ['img-2'])  # type: ignore[arg-type]

    assert order == ["release ['img-2']", 'publish']
    assert service.announced == [[('a', '/p/1.jpg'), ('b', '/p/1.jpg'), ('c', '/p/3.jpg')]]


@pytest.mark.asyncio
async def test_finish_commit_with_nothing_to_do_touches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def release(_ids: Any) -> int:
        raise AssertionError('nothing to release')

    monkeypatch.setattr(content_key, 'release_bound', release)
    service = _Publisher()
    await ingest_index.finish_commit(service, [_outcome([], status='failed')], [])  # type: ignore[arg-type]
    assert service.announced == []


@pytest.mark.asyncio
async def test_a_failed_release_does_not_skip_the_announcement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def release(_ids: Any) -> int:
        raise PoolTimeoutError('QueuePool limit reached')

    monkeypatch.setattr(content_key, 'release_bound', release)
    service = _Publisher()
    with pytest.raises(PoolTimeoutError):
        await ingest_index.finish_commit(service, [_outcome(['a'])], ['img-1'])  # type: ignore[arg-type]
    assert service.announced == [[('a', '/p/x.jpg')]]


@pytest.mark.asyncio
async def test_a_failed_announcement_after_a_failed_release_carries_both_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def release(_ids: Any) -> int:
        raise PoolTimeoutError('QueuePool limit reached')

    monkeypatch.setattr(content_key, 'release_bound', release)
    with pytest.raises(RuntimeError, match='publish failed') as raised:
        await ingest_index.finish_commit(_Publisher(fail=True), [_outcome(['a'])], ['img-1'])  # type: ignore[arg-type]
    assert isinstance(raised.value.__context__, PoolTimeoutError)


def _crop_ids(captured_events: list[dict[str, Any]]) -> list[str]:
    return sorted(e['crop_id'] for e in captured_events if e['type'] == 'crop.created')


@pytest.mark.asyncio
async def test_one_failed_write_stops_no_other_photo_and_the_finish_still_runs(
    captured_events: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    released: list[list[str]] = []

    async def release(image_ids: Any) -> int:
        released.append(sorted(image_ids))
        return 1

    monkeypatch.setattr(content_key, 'release_bound', release)
    svc, os_fake, _ = _make_service(detections=[(0.1, 0.1, 0.5, 0.5, 0.9, 1)])
    real_bulk = svc._bulk_index

    async def bulk(image_doc: Any, crop_docs: Any, *a: Any, **k: Any) -> Any:
        if image_doc is not None and image_doc['image_path'] == '/p/1.jpg':
            raise IngestWriteError('write refused', release_claim=True)
        return await real_bulk(image_doc, crop_docs, *a, **k)

    monkeypatch.setattr(svc, '_bulk_index', bulk)
    images = [_jpeg_bytes(seed=i + 1) for i in range(3)]
    result = await svc.ingest_batch(images, [f'/p/{i}.jpg' for i in range(3)])

    assert [r.status for r in result.results] == ['success', 'failed', 'success']
    assert released == [[content_image_id(content_imohash(images[1]))]]
    stored = {i['image_path'] for i in os_fake.images.values()}
    assert stored == {'/p/0.jpg', '/p/2.jpg'}
    events = [e for e in captured_events if e['type'] == 'crop.created']
    assert sorted({e['image_path'] for e in events}) == ['/p/0.jpg', '/p/2.jpg']  # none for photo 1
    assert _crop_ids(captured_events) == sorted(os_fake.items)


@pytest.mark.asyncio
async def test_a_cancelled_commit_still_releases_and_announces_what_it_stored(
    captured_events: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Photo 0 is stored, photo 1's write failed (its claim is to be released), photo 2's write
    hangs; the request is cancelled. The finish runs: photo 0 is announced, photo 1 released."""
    released: list[list[str]] = []

    async def release(image_ids: Any) -> int:
        released.append(sorted(image_ids))
        return 1

    monkeypatch.setattr(content_key, 'release_bound', release)
    svc, os_fake, _ = _make_service(detections=[(0.1, 0.1, 0.5, 0.5, 0.9, 1)])
    real_bulk = svc._bulk_index
    stored, failed, hung = asyncio.Event(), asyncio.Event(), asyncio.Event()
    real_cache = ingest_index._cache_crops

    async def cache(*a: Any, **k: Any) -> None:
        await real_cache(*a, **k)
        stored.set()  # photo 0's write_prepared returns, and is recorded, before anything yields

    async def bulk(image_doc: Any, crop_docs: Any, *a: Any, **k: Any) -> Any:
        path = image_doc['image_path']
        if path == '/p/1.jpg':
            failed.set()
            raise IngestWriteError('write refused', release_claim=True)
        if path == '/p/2.jpg':
            await asyncio.wait_for(asyncio.gather(stored.wait(), failed.wait()), 10)
            hung.set()
            await asyncio.Event().wait()
        return await real_bulk(image_doc, crop_docs, *a, **k)

    monkeypatch.setattr(ingest_index, '_cache_crops', cache)
    monkeypatch.setattr(svc, '_bulk_index', bulk)
    images = [_jpeg_bytes(seed=i + 1) for i in range(3)]
    task = asyncio.create_task(svc.ingest_batch(images, [f'/p/{i}.jpg' for i in range(3)]))
    await asyncio.wait_for(hung.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert released == [[content_image_id(content_imohash(images[1]))]]
    assert {i['image_path'] for i in os_fake.images.values()} == {'/p/0.jpg'}
    assert _crop_ids(captured_events) == sorted(os_fake.items)
    assert _crop_ids(captured_events) != []


# index_items: a batch of one, also for the callers that bring their own items (plan 13a F25).


async def _stored_image(svc: Any, seed: int) -> bytes:
    data = _jpeg_bytes(seed=seed)
    assert (await svc.ingest_one(data, f'/lib/{seed}.jpg')).status == 'success'
    return data


@pytest.mark.asyncio
async def test_a_reprocess_adds_items_to_a_stored_image_without_a_claim(
    captured_events: list[dict[str, Any]], content_rows: Any, pending_calls: Any
) -> None:
    """``created`` False (re-detection, open-vocabulary run): items and their events, no images
    doc, no claim, no image-level pending row."""
    svc, os_fake, _ = _make_service(detections=[(0.05, 0.05, 0.4, 0.4, 0.9, 1)])
    data = await _stored_image(svc, 31)
    (image_id,) = os_fake.images
    before = dict(os_fake.images[image_id])
    rows_before = dict(content_rows.rows)
    old_items = set(os_fake.items)
    captured_events.clear()
    pending_calls.calls.clear()

    ctx = await svc.ingest_image(data, '/lib/31.jpg', 'reprocess', adopt_existing=True)
    assert ctx.created is False
    outcome = await svc.index_items(
        ctx, [DetectedItem(bbox_pixel=(100.0, 100.0, 160.0, 150.0), score=0.8)]
    )

    assert outcome.result.status == 'success'
    (new_id,) = set(os_fake.items) - old_items
    assert outcome.created_ids == [new_id]
    assert os_fake.images[image_id] == before  # the images doc is untouched
    assert content_rows.rows == rows_before  # no claim taken
    assert pending_calls.calls == [('item', [new_id], 'added')]
    assert [e['crop_id'] for e in captured_events if e['type'] == 'crop.created'] == [new_id]


@pytest.mark.asyncio
async def test_an_import_creates_the_image_with_its_split_and_announces_its_items(
    captured_events: list[dict[str, Any]], content_rows: Any, pending_calls: Any
) -> None:
    """A dataset import (``seed_region`` False, the images doc stamped by the importer): a claim,
    the images doc and the items, both pending levels, one event per item."""
    svc, os_fake, _ = _make_service(detections=[])
    data = _jpeg_bytes(seed=32)
    ctx = await svc.ingest_image(data, '/imp/32.jpg', 'import', adopt_existing=True)
    assert ctx.created is True
    ctx.image_extra = {'dataset_split': 'val'}
    items = [
        DetectedItem(bbox_pixel=(10.0, 10.0, 60.0, 50.0), score=1.0),
        DetectedItem(bbox_pixel=(70.0, 20.0, 120.0, 90.0), score=1.0),
    ]

    outcome = await svc.index_items(ctx, items, seed_region=False)

    assert outcome.result.status == 'success'
    assert outcome.result.n_crops == 2
    (image,) = os_fake.images.values()
    assert image['dataset_split'] == 'val'
    assert len(os_fake.items) == 2
    assert content_rows.rows == {ctx.imohash: ctx.image_id}
    assert sorted(pending_calls.calls) == sorted(
        [('item', sorted(outcome.created_ids), 'added'), ('image', [ctx.image_id], 'added')]
    )
    assert _crop_ids(captured_events) == sorted(outcome.created_ids)


@pytest.mark.asyncio
async def test_index_items_of_a_photo_whose_content_another_image_holds_writes_nothing(
    captured_events: list[dict[str, Any]], content_rows: Any
) -> None:
    svc, os_fake, _ = _make_service(detections=[])
    data = _jpeg_bytes(seed=33)
    ctx = await svc.ingest_image(data, '/imp/33.jpg', 'import', adopt_existing=True)
    content_rows.rows[ctx.imohash] = 'someone-else'
    os_fake.images['someone-else'] = {'image_id': 'someone-else', 'item_ids': []}

    outcome = await svc.index_items(
        ctx, [DetectedItem(bbox_pixel=(10.0, 10.0, 60.0, 50.0), score=1.0)]
    )

    assert outcome.result.status == 'duplicate'
    assert outcome.result.image_id == 'someone-else'
    assert list(os_fake.items) == []
    assert _crop_ids(captured_events) == []
