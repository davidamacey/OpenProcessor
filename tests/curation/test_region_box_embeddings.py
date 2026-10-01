"""Per-box region embeddings: the pure join/merge rules and the OCC write.

``region_box_embeddings`` is a sibling nested list keyed by ``box_id``; the
vector records the geometry it was computed from so a box a human moved is
recognised as stale, an entry whose box is gone is ignored and pruned, and
a box write never has to touch the vectors.
"""

from __future__ import annotations

from typing import Any

import pytest

from curation.test_regions_router import _FakeRegionOS
from src.config import get_region_fields
from src.services.curation.region_box_embeddings import (
    current_vectors,
    entry_for,
    join_box_vectors,
    merge_box_embeddings,
    missing_boxes,
    write_box_embeddings,
)
from src.services.curation.region_boxes import RegionBox, boxes_write_fields


F = get_region_fields()
INDEX = 'items'


def _box(box_id: str, state: str = 'accepted', x: float = 0.1) -> RegionBox:
    return RegionBox(box_id=box_id, bbox_norm=(x, 0.1, x + 0.2, 0.4), state=state)


def _doc(boxes: list[RegionBox], entries: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    doc = {'crop_id': 'c1', **boxes_write_fields(boxes, current_src={})}
    if entries is not None:
        doc[F.box_embeddings] = entries
    return doc


def test_merge_replaces_prunes_orphans_and_stale_entries_and_keeps_the_rest() -> None:
    b1, b2, b3, moved = _box('b1'), _box('b2', x=0.5), _box('b3', x=0.6), _box('b4', x=0.7)
    existing = [
        entry_for(b1, [1.0]),
        entry_for(b2, [2.0]),
        entry_for(_box('gone', x=0.9), [9.0]),
        entry_for(_box('b4', x=0.1), [4.0]),  # b4 has since moved
    ]
    new = [entry_for(b2, [22.0]), entry_for(b3, [3.0])]

    merged = merge_box_embeddings(existing, new, live=[b1, b2, b3, moved])

    assert merged == [entry_for(b1, [1.0]), entry_for(b2, [22.0]), entry_for(b3, [3.0])]


def test_merge_never_stores_a_new_entry_for_a_box_that_is_gone_or_moved() -> None:
    live = _box('b1', x=0.3)
    stale = entry_for(_box('b1', x=0.1), [1.0])
    assert merge_box_embeddings(None, [entry_for(_box('b9'), [1.0]), stale], live=[live]) == []


def test_current_vectors_skips_orphans_and_stale_entries() -> None:
    b1, b2 = _box('b1'), _box('b2', x=0.5)
    moved_b2 = _box('b2', x=0.6)
    doc = _doc(
        [b1, moved_b2],
        [
            entry_for(b1, [1.0]),
            entry_for(b2, [2.0]),  # computed from b2's OLD geometry
            entry_for(_box('gone'), [9.0]),
        ],
    )

    assert current_vectors(doc) == {'b1': [1.0]}


def test_join_pairs_each_box_with_its_vector_in_box_order() -> None:
    b1, b2 = _box('b1'), _box('b2', x=0.5)
    doc = _doc([b1, b2], [entry_for(b2, [2.0]), entry_for(b1, [1.0])])

    assert [(box.box_id, vec) for box, vec in join_box_vectors(doc)] == [
        ('b1', [1.0]),
        ('b2', [2.0]),
    ]


def test_missing_boxes_are_the_embeddable_ones_without_a_valid_vector() -> None:
    accepted = _box('b1')
    fp = _box('b2', 'false_positive', 0.4)
    rejected = _box('b3', 'rejected', 0.6)
    proposed = _box('b4', 'proposed', 0.7)
    moved = _box('b5', x=0.8)
    stale = entry_for(_box('b5', x=0.1), [5.0])
    doc = _doc([accepted, fp, rejected, proposed, moved], [stale])

    assert [b.box_id for b in missing_boxes(doc)] == ['b1', 'b2', 'b5']
    # An _source that carries only box_id / bbox_norm (no vectors) says the same.
    light = {**doc, F.box_embeddings: [{k: v for k, v in stale.items() if k != 'embedding'}]}
    assert [b.box_id for b in missing_boxes(light)] == ['b1', 'b2', 'b5']


@pytest.mark.asyncio
async def test_write_merges_with_stored_entries_and_never_touches_the_box_list() -> None:
    b1, b2 = _box('b1'), _box('b2', x=0.5)
    doc = _doc([b1, b2], [entry_for(b1, [1.0])])
    client = _FakeRegionOS({'c1': doc})
    boxes_before = list(client._docs['c1'][F.boxes])

    counts = await write_box_embeddings(client, index=INDEX, by_crop={'c1': [entry_for(b2, [2.0])]})

    assert counts == {'written': 1, 'unchanged': 0, 'skipped': 0, 'errors': 0}
    stored = client._docs['c1']
    assert [e['box_id'] for e in stored[F.box_embeddings]] == ['b1', 'b2']
    assert stored[F.boxes] == boxes_before
    assert stored[F.revision] == doc[F.revision]  # an embedding write is not an edit


@pytest.mark.asyncio
async def test_write_drops_a_vector_for_a_box_deleted_in_the_meantime() -> None:
    b1 = _box('b1')
    client = _FakeRegionOS({'c1': _doc([b1])})

    await write_box_embeddings(
        client,
        index=INDEX,
        by_crop={'c1': [entry_for(b1, [1.0]), entry_for(_box('b2', x=0.5), [2.0])]},
    )

    assert [e['box_id'] for e in client._docs['c1'][F.box_embeddings]] == ['b1']


@pytest.mark.asyncio
async def test_write_retries_after_a_version_conflict() -> None:
    b1 = _box('b1')
    client = _FakeRegionOS({'c1': _doc([b1])})
    real_bulk = client.bulk
    calls = {'n': 0}

    async def _racing_bulk(*, body: list[dict[str, Any]], refresh: Any = False) -> dict[str, Any]:
        calls['n'] += 1
        if calls['n'] == 1:
            client._seq['c1'] += 1  # a concurrent writer got in first
        return await real_bulk(body=body, refresh=refresh)

    client.bulk = _racing_bulk  # type: ignore[method-assign]

    counts = await write_box_embeddings(client, index=INDEX, by_crop={'c1': [entry_for(b1, [1.0])]})

    assert counts['written'] == 1
    assert calls['n'] == 2
    assert client._docs['c1'][F.box_embeddings][0]['embedding'] == [1.0]


@pytest.mark.asyncio
async def test_write_for_a_missing_item_is_skipped() -> None:
    client = _FakeRegionOS({})
    counts = await write_box_embeddings(
        client, index=INDEX, by_crop={'nope': [entry_for(_box('b1'), [1.0])]}
    )
    assert counts == {'written': 0, 'unchanged': 0, 'skipped': 1, 'errors': 0}


@pytest.mark.asyncio
async def test_prune_drops_the_entry_of_a_moved_and_of_a_deleted_box() -> None:
    from src.services.curation.region_box_embeddings import prune_box_embeddings

    old1, b2, b3 = _box('b1'), _box('b2', x=0.5), _box('b3', x=0.7)
    moved1 = _box('b1', x=0.2)
    doc = _doc([moved1, b2], [entry_for(old1, [1.0]), entry_for(b2, [2.0]), entry_for(b3, [3.0])])
    client = _FakeRegionOS({'c1': doc})

    await prune_box_embeddings(client, index=INDEX, crop_ids=['c1'])

    assert [e['box_id'] for e in client._docs['c1'][F.box_embeddings]] == ['b2']


@pytest.mark.asyncio
async def test_prune_with_nothing_to_drop_writes_nothing() -> None:
    from src.services.curation.region_box_embeddings import prune_box_embeddings

    b1 = _box('b1')
    client = _FakeRegionOS({'c1': _doc([b1], [entry_for(b1, [1.0])])})
    seq = dict(client._seq)

    await prune_box_embeddings(client, index=INDEX, crop_ids=['c1'])

    assert client._seq == seq


@pytest.mark.asyncio
async def test_prune_failure_is_logged_not_raised() -> None:
    from src.services.curation.region_box_embeddings import prune_box_embeddings

    class _Boom:
        async def mget(self, **_kw: Any) -> dict[str, Any]:
            raise RuntimeError('down')

    await prune_box_embeddings(_Boom(), index=INDEX, crop_ids=['c1'])
