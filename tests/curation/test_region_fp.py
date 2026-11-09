"""Region boxes as rows (read, guarded write) and the false-positive matcher: sub-typing and the
auto FP pull.

Fixtures are production-shaped items (``boxes_write_fields`` box lists and
``region_box_embeddings`` entries) in the query-evaluating fake, so what is
asserted is what lands on the stored boxes -- including a write that races a human edit.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import numpy as np
import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_region_fields
from src.domain.region_boxes import RegionBox, boxes_write_fields, read_boxes
from src.services.clustering import region_fp as rbc
from src.services.curation.clustering.region_box_rows import (
    box_filter,
    scroll_box_rows,
    write_box_edits,
)
from src.services.curation.region_box_embeddings import entry_for


F = get_region_fields()
INDEX = 'op_items'


@pytest.fixture(autouse=True)
def _items_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rbc, 'items_index', lambda: INDEX)


class _OS(QueryFakeOpenSearch):
    """The shared fake plus two hooks: ``after_scan`` runs when a scroll is
    cleared (the rows were read, the model is fitting) and ``before_bulk``
    just before a bulk write (after the OCC read that fetched its version)
    -- a writer racing the fit / the write."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.after_scan: Any = None
        self.before_bulk: Any = None

    async def bulk(self, **kw: Any) -> dict[str, Any]:
        if self.before_bulk is not None:
            hook, self.before_bulk = self.before_bulk, None
            hook(self)
        return await super().bulk(**kw)

    async def clear_scroll(self, **kw: Any) -> dict[str, Any]:
        if self.after_scan is not None:
            hook, self.after_scan = self.after_scan, None
            hook(self)
        return await super().clear_scroll(**kw)


def _vec(*xyz: float) -> list[float]:
    v = np.asarray(xyz, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


A = (1.0, 0.0, 0.0)
B = (0.0, 1.0, 0.0)


def _box(box_id: str, state: str = 'accepted', x: float = 0.1, **over: Any) -> RegionBox:
    return RegionBox(box_id=box_id, bbox_norm=(x, 0.1, x + 0.2, 0.4), state=state, **over)


def _item(
    crop_id: str,
    boxes: list[tuple[RegionBox, tuple[float, float, float] | None]],
    **extra: Any,
) -> dict[str, Any]:
    stored = [b for b, _v in boxes]
    doc: dict[str, Any] = {
        'crop_id': crop_id,
        'class_name': 'thing',
        **boxes_write_fields(stored, current_src={}),
        F.status: 'detected',
        **extra,
    }
    doc[F.box_embeddings] = [entry_for(b, _vec(*v)) for b, v in boxes if v is not None]
    return doc


def _jitter(base: tuple[float, float, float], i: int) -> tuple[float, float, float]:
    return (base[0] + 0.001 * i, base[1] + 0.001 * i, base[2] + 0.0005 * i)


def _bulk_items(n_per_group: int = 20) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for i in range(n_per_group):
        docs[f'a{i}'] = _item(f'a{i}', [(_box('b1'), _jitter(A, i))])
        docs[f'b{i}'] = _item(f'b{i}', [(_box('b1'), _jitter(B, i))])
    return docs


def _boxes(os_: QueryFakeOpenSearch, crop_id: str) -> dict[str, RegionBox]:
    return {b.box_id: b for b in read_boxes(os_.docs(INDEX)[crop_id], F)}


# ---------------------------------------------------------------------------
# guarded writes
# ---------------------------------------------------------------------------


def _to_fp(box: RegionBox) -> RegionBox:
    return dataclasses.replace(box, state='false_positive')


@pytest.mark.asyncio
async def test_a_write_racing_a_human_edit_is_re_merged_not_dropped_or_clobbering() -> None:
    docs = _bulk_items()
    os_ = _OS({INDEX: docs})
    human_geometry = (0.55, 0.55, 0.75, 0.85)

    def human_moves_a0(fake: QueryFakeOpenSearch) -> None:
        doc = fake.docs(INDEX)['a0']
        moved = dataclasses.replace(
            read_boxes(doc, F)[0], bbox_norm=human_geometry, source='human', detector='human'
        )
        doc.update(boxes_write_fields([moved], current_src=doc))
        fake._bump(INDEX, 'a0')

    os_.after_scan = human_moves_a0
    rows = await scroll_box_rows(os_, index=INDEX, states=('accepted',))
    edits = {r.crop_id: {r.box_id: _to_fp} for r in rows}

    await write_box_edits(os_, index=INDEX, edits=edits, respect_human=True)

    box = _boxes(os_, 'a0')['b1']
    # The human's box is kept whole -- it became locked while the rows were read.
    assert box.bbox_norm == human_geometry
    assert box.source == 'human'
    assert box.state == 'accepted'
    # ... and the conflict didn't cost any other item its write.
    assert _boxes(os_, 'a1')['b1'].state == 'false_positive'


@pytest.mark.asyncio
async def test_a_version_conflict_on_the_write_is_re_merged_against_the_new_version() -> None:
    docs = _bulk_items()
    os_ = _OS({INDEX: docs})

    def another_writer(fake: QueryFakeOpenSearch) -> None:
        # Between the OCC read and the conditional write another writer
        # bumps the doc: the write 409s and must be re-merged, not dropped.
        fake.docs(INDEX)['a0']['updated_at'] = 'later'
        fake._bump(INDEX, 'a0')

    os_.before_bulk = another_writer

    await write_box_edits(os_, index=INDEX, edits={'a0': {'b1': _to_fp}}, respect_human=True)

    assert _boxes(os_, 'a0')['b1'].state == 'false_positive'
    assert os_.docs(INDEX)['a0']['updated_at'] != 'later'  # the retried write landed


@pytest.mark.asyncio
async def test_write_box_edits_leaves_human_final_items_and_locked_boxes_alone() -> None:
    docs = {
        'verified': _item('verified', [(_box('b1'), A)], **{F.verifier: 'human'}),
        'validated': _item('validated', [(_box('b1'), A)], **{F.validated: True}),
        'own': _item('own', [(_box('b1', source='human'), A), (_box('b2', x=0.5), B)]),
    }
    os_ = _OS({INDEX: docs})

    await write_box_edits(
        os_,
        index=INDEX,
        edits={
            'verified': {'b1': _to_fp},
            'validated': {'b1': _to_fp},
            'own': {'b1': _to_fp, 'b2': _to_fp},
        },
        respect_human=True,
    )

    assert _boxes(os_, 'verified')['b1'].state == 'accepted'
    assert _boxes(os_, 'validated')['b1'].state == 'accepted'
    own = _boxes(os_, 'own')
    assert own['b1'].state == 'accepted'  # a human's box
    assert own['b2'].state == 'false_positive'  # its machine sibling is edited


# ---------------------------------------------------------------------------
# rows + counts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rows_are_the_boxes_in_the_asked_states_with_a_vector() -> None:
    item = _item(
        'x',
        [
            (_box('b1'), A),
            (_box('b2', 'false_positive', x=0.5), B),
            (_box('b3', x=0.7), None),
        ],
    )
    os_ = _OS({INDEX: {'x': item}})

    accepted = await scroll_box_rows(os_, index=INDEX, states=('accepted',))
    fps = await scroll_box_rows(os_, index=INDEX, states=('false_positive',))

    assert [(r.crop_id, r.box_id) for r in accepted] == [('x', 'b1')]
    assert [(r.crop_id, r.box_id) for r in fps] == [('x', 'b2')]


def test_box_filter_selects_the_state_on_the_same_box() -> None:
    from curation.query_fakes import matches

    item = _item('x', [(_box('b1'), A), (_box('b2', 'false_positive', x=0.5), B)])

    assert matches(item, box_filter(('accepted',)))
    assert matches(item, box_filter(('false_positive',)))
    assert not matches(item, box_filter(('rejected',)))


@pytest.mark.asyncio
async def test_write_box_edits_applies_only_the_named_boxes() -> None:
    item = _item('x', [(_box('b1'), A), (_box('b2', x=0.5), B)])
    os_ = _OS({INDEX: {'x': item}})

    result = await write_box_edits(
        os_,
        index=INDEX,
        edits={'x': {'b2': _to_fp}},
        respect_human=True,
    )

    assert result['items_written'] == 1
    assert result['boxes_changed'] == 1
    boxes = _boxes(os_, 'x')
    assert boxes['b1'].state == 'accepted'
    assert boxes['b2'].state == 'false_positive'


@pytest.mark.asyncio
async def test_write_box_edits_counts_boxes_and_items_separately() -> None:
    docs = {
        'x': _item('x', [(_box('b1'), A), (_box('b2', x=0.5), B)]),
        'y': _item('y', [(_box('b1'), A)]),
        'z': _item('z', [(_box('b1', 'false_positive'), A)]),
    }
    os_ = _OS({INDEX: docs})

    to_three = _to_fp

    result = await write_box_edits(
        os_,
        index=INDEX,
        edits={
            'x': {'b1': to_three, 'b2': to_three},
            'y': {'b1': to_three},
            'z': {'b1': to_three},
        },
        respect_human=True,
    )

    assert result == {
        'items_written': 2,
        'boxes_changed': 3,
        'items_unchanged': 1,
        'items_conflicted': 0,
        'items_errored': 0,
    }


@pytest.mark.asyncio
async def test_a_box_edit_advances_the_revision_an_editor_holds() -> None:
    docs = {'x': _item('x', [(_box('b1'), A), (_box('b2', x=0.5), B)], **{F.revision: 7})}
    os_ = _OS({INDEX: docs})

    await write_box_edits(os_, index=INDEX, edits={'x': {'b1': _to_fp}}, respect_human=True)

    assert os_.docs(INDEX)['x'][F.revision] == 8


# ---------------------------------------------------------------------------
# FP centroids + auto pull
# ---------------------------------------------------------------------------


@pytest.fixture
def fp_store_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr('src.services.detection.fp_store.fp_store_dir', lambda *_a, **_k: tmp_path)
    return tmp_path


@pytest.mark.asyncio
async def test_fp_centroids_are_sub_typed_from_fp_boxes_only(fp_store_dir: Any) -> None:
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    fp1 = _box('b1', 'false_positive')
    fp2 = _box('b2', 'false_positive', x=0.5)
    docs = {
        # a good box (vector along A) beside an FP box (vector along B)
        'x': _item('x', [(_box('b1'), A), (fp2, B)]),
        'y': _item('y', [(fp1, B)]),
        'z': _item('z', [(_box('b1'), A)]),
    }
    os_ = _OS({INDEX: docs})

    result = await rbc.build_region_fp_centroids(os_)

    assert result['status'] == 'success'
    assert result['n_boxes'] == 2  # the two FP boxes; the good boxes don't count
    assert result['k'] == 1
    store = FalsePositiveCentroidStore()
    assert store.load()
    centroid = np.asarray(store._index.reconstruct(0))
    assert centroid == pytest.approx(np.asarray(_vec(*B)), abs=1e-5)
    assert store.metadata['subids'] == ['a']
    assert store.metadata['n_boxes'] == 2
    # Nothing is written onto the boxes: the sub-type labels live in the store.
    assert [b.state for b in _boxes(os_, 'x').values()] == ['accepted', 'false_positive']


@pytest.mark.asyncio
async def test_fp_centroids_without_fp_boxes_is_a_noop(fp_store_dir: Any) -> None:
    os_ = _OS({INDEX: {'z': _item('z', [(_box('b1'), A)])}})

    result = await rbc.build_region_fp_centroids(os_)

    assert result == {'status': 'skipped', 'reason': 'no_fp_embeddings', 'n_boxes': 0}


def _save_centroid(fp_store_dir: Any, vector: tuple[float, float, float]) -> None:
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    FalsePositiveCentroidStore().save(
        np.asarray([_vec(*vector)], dtype=np.float32),
        {'trained_at': 't', 'k': 1, 'n_boxes': 1, 'subids': ['a'], 'dim': 3},
    )


@pytest.mark.asyncio
async def test_auto_pull_flips_the_matching_box_and_rederives_the_item(
    fp_store_dir: Any,
) -> None:
    _save_centroid(fp_store_dir, B)
    docs = {
        'only': _item('only', [(_box('b1'), B)]),
        'sibling': _item('sibling', [(_box('b1'), A), (_box('b2', x=0.5), B)]),
        'far': _item('far', [(_box('b1'), A)]),
        'human': _item('human', [(_box('b1'), B)], **{F.verifier: 'human'}),
        'own': _item('own', [(_box('b1', source='human'), B)]),
        'holdout': _item('holdout', [(_box('b1'), B)], test_holdout=True),
    }
    os_ = _OS({INDEX: docs})

    result = await rbc.auto_assign_fp_from_centroids(os_, threshold=0.2)

    assert result['n_boxes_moved'] == 2
    only = os_.docs(INDEX)['only']
    box = _boxes(os_, 'only')['b1']
    assert box.state == 'false_positive'
    assert only[F.status] == 'false_positive'
    assert only[F.verified] is False
    assert only[F.label_source] == 'auto_fp_centroid'
    # A sibling accepted box keeps the item `detected`; only the matching box moves.
    sibling = os_.docs(INDEX)['sibling']
    boxes = _boxes(os_, 'sibling')
    assert boxes['b1'].state == 'accepted'
    assert boxes['b2'].state == 'false_positive'
    assert sibling[F.status] == 'detected'
    assert sibling[F.verified] is True
    assert sibling[F.count] == 1
    for untouched in ('far', 'human', 'own', 'holdout'):
        assert all(b.state == 'accepted' for b in _boxes(os_, untouched).values()), untouched
        assert os_.docs(INDEX)[untouched][F.revision] == 1, untouched


@pytest.mark.asyncio
async def test_auto_pull_never_flips_a_box_that_is_no_longer_accepted(
    fp_store_dir: Any,
) -> None:
    _save_centroid(fp_store_dir, B)
    os_ = _OS({INDEX: {'a': _item('a', [(_box('b1'), B), (_box('b2', x=0.5), A)])}})

    def verifier_rejects_b1(fake: QueryFakeOpenSearch) -> None:
        doc = fake.docs(INDEX)['a']
        boxes = [
            dataclasses.replace(b, state='rejected', rejection_reason='verifier_rejected')
            if b.box_id == 'b1'
            else b
            for b in read_boxes(doc, F)
        ]
        doc.update(boxes_write_fields(boxes, current_src=doc))
        fake._bump(INDEX, 'a')

    os_.after_scan = verifier_rejects_b1

    await rbc.auto_assign_fp_from_centroids(os_, threshold=0.2)

    boxes = _boxes(os_, 'a')
    assert boxes['b1'].state == 'rejected'


@pytest.mark.asyncio
async def test_auto_pull_without_centroids_is_a_noop(fp_store_dir: Any) -> None:
    os_ = _OS({INDEX: {'a': _item('a', [(_box('b1'), B)])}})

    result = await rbc.auto_assign_fp_from_centroids(os_)

    assert result['status'] == 'skipped'
    assert _boxes(os_, 'a')['b1'].state == 'accepted'


@pytest.mark.asyncio
async def test_fp_candidate_pool_is_accepted_unlocked_boxes_with_vectors() -> None:
    docs = {
        'ok': _item('ok', [(_box('b1'), A)]),
        'fp': _item('fp', [(_box('b1', 'false_positive'), A)]),
        'rejected': _item('rejected', [(_box('b1', 'rejected'), A)]),
        'no_vec': _item('no_vec', [(_box('b1'), None)]),
        'hold': _item('hold', [(_box('b1'), A)], test_holdout=True),
        'human': _item('human', [(_box('b1'), A)], **{F.label_source: 'human'}),
        'own': _item('own', [(_box('b1', source='human'), A)]),
    }
    os_ = _OS({INDEX: docs})

    rows = await rbc.fp_candidate_rows(os_)

    assert [(r.crop_id, r.box_id) for r in rows] == [('ok', 'b1')]


# ---------------------------------------------------------------------------
# read order (a fit is a function of the box set, not the read order)
# ---------------------------------------------------------------------------


def _random_box_items(n: int = 120) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(7)
    centres = rng.standard_normal((6, 8))
    docs: dict[str, dict[str, Any]] = {}
    for i in range(n):
        v = centres[i % 6] + 0.9 * rng.standard_normal(8)
        two = [(_box('b1'), tuple(v[:3] + 4.0)), (_box('a0', x=0.5), tuple(v[3:6] + 4.0))]
        docs[f'c{i:03d}'] = _item(f'c{i:03d}', two)
    return docs


def _permuted_insertion(docs: dict[str, dict[str, Any]], seed: int) -> dict[str, dict[str, Any]]:
    order = np.random.default_rng(seed).permutation(len(docs))
    keys = list(docs)
    return {keys[i]: copy.deepcopy(docs[keys[i]]) for i in order}


@pytest.mark.asyncio
async def test_scroll_box_rows_come_back_sorted_by_crop_then_box_id() -> None:
    os_ = _OS({INDEX: _permuted_insertion(_random_box_items(), 1)})

    rows = await scroll_box_rows(os_, index=INDEX, states=('accepted',))

    ids = [(r.crop_id, r.box_id) for r in rows]
    assert ids == sorted(ids)
    assert len(ids) == 240
