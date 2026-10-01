"""Region clustering over BOXES: partition, refine, FP sub-typing and the
auto FP pull.

Fixtures are production-shaped items (``boxes_write_fields`` box lists and
``region_box_embeddings`` entries) in the query-evaluating fake, so what is
asserted is what lands on the stored boxes -- including a box that shares
an item with a box in another cluster, and a write that races a human edit.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import numpy as np
import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_region_fields
from src.services.curation.cluster_ids import FALSE_POSITIVE_REGION_CLUSTER_ID
from src.services.curation.clustering import region_box_clustering as rbc
from src.services.curation.clustering.region_box_rows import (
    box_filter,
    count_boxes,
    scroll_box_rows,
    write_box_edits,
)
from src.services.curation.region_box_embeddings import entry_for
from src.services.curation.region_boxes import RegionBox, boxes_write_fields, read_boxes


pytestmark = pytest.mark.asyncio

F = get_region_fields()
INDEX = 'op_items'
FP = FALSE_POSITIVE_REGION_CLUSTER_ID


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
# partition
# ---------------------------------------------------------------------------


async def test_partition_writes_the_cluster_onto_each_box_of_an_item() -> None:
    docs = _bulk_items()
    # One item with a box in each direction: its boxes land in different buckets.
    docs['two'] = _item('two', [(_box('b1'), A), (_box('b2', x=0.5), B)])
    os_ = _OS({INDEX: docs})

    result = await rbc.cluster_region_residuals(os_)

    assert result['status'] == 'success'
    assert result['n_boxes'] == 42
    # Counts name their unit: 42 boxes live in 41 items.
    assert result['n_boxes_changed'] == 42
    assert result['n_items_written'] == 41
    two = _boxes(os_, 'two')
    assert two['b1'].cluster_id is not None
    assert two['b2'].cluster_id is not None
    assert two['b1'].cluster_id != two['b2'].cluster_id
    group_a = {_boxes(os_, f'a{i}')['b1'].cluster_id for i in range(20)}
    group_b = {_boxes(os_, f'b{i}')['b1'].cluster_id for i in range(20)}
    assert group_a.isdisjoint(group_b)
    assert two['b1'].cluster_id in group_a
    assert two['b2'].cluster_id in group_b
    assert 'region_cluster_id' not in os_.docs(INDEX)['two']
    # The partition changes placement only, so the revision is the seeded one.
    assert os_.docs(INDEX)['two'][F.revision] == 1


async def test_partition_never_reshuffles_false_positive_boxes_or_uses_their_vectors() -> None:
    docs = _bulk_items()
    fp_box = _box('b2', 'false_positive', x=0.5, cluster_id=FP, cluster_distance=0.0)
    docs['mixed'] = _item('mixed', [(_box('b1'), A), (fp_box, B)])
    os_ = _OS({INDEX: docs})

    result = await rbc.cluster_region_residuals(os_)

    assert result['n_boxes'] == 41  # the FP box is not a row
    mixed = _boxes(os_, 'mixed')
    assert mixed['b2'].cluster_id == FP
    assert mixed['b2'].cluster_distance == 0.0
    assert mixed['b1'].cluster_id not in (None, FP)


async def test_partition_skips_a_box_whose_vector_is_stale() -> None:
    docs = _bulk_items()
    moved = _box('b1', x=0.6)
    stale = _item('stale', [(_box('b1'), A)])
    stale[F.boxes] = [moved.to_doc()]  # the box moved after its vector was computed
    docs['stale'] = stale
    os_ = _OS({INDEX: docs})

    result = await rbc.cluster_region_residuals(os_)

    assert result['n_boxes'] == 40
    assert _boxes(os_, 'stale')['b1'].cluster_id is None


async def test_partition_leaves_human_final_items_and_locked_boxes_alone() -> None:
    docs = _bulk_items()
    docs['verified'] = _item('verified', [(_box('b1'), A)], **{F.verifier: 'human'})
    docs['validated'] = _item('validated', [(_box('b1'), A)], **{F.validated: True})
    docs['own'] = _item('own', [(_box('b1', source='human'), A), (_box('b2', x=0.5), B)])
    os_ = _OS({INDEX: docs})

    await rbc.cluster_region_residuals(os_)

    assert _boxes(os_, 'verified')['b1'].cluster_id is None
    assert _boxes(os_, 'validated')['b1'].cluster_id is None
    own = _boxes(os_, 'own')
    assert own['b1'].cluster_id is None  # a human's box
    assert own['b2'].cluster_id is not None  # its machine sibling is clustered


async def test_partition_below_the_minimum_is_a_noop() -> None:
    os_ = _OS({INDEX: {'a': _item('a', [(_box('b1'), A)])}})

    result = await rbc.cluster_region_residuals(os_)

    assert result['status'] == 'skipped'
    assert _boxes(os_, 'a')['b1'].cluster_id is None


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

    await rbc.cluster_region_residuals(os_)

    box = _boxes(os_, 'a0')['b1']
    # The human's box is kept whole -- it became locked while the model fit.
    assert box.bbox_norm == human_geometry
    assert box.source == 'human'
    assert box.cluster_id is None
    # ... and the conflict didn't cost any other item its write.
    assert _boxes(os_, 'a1')['b1'].cluster_id is not None


async def test_a_version_conflict_on_the_write_is_re_merged_against_the_new_version() -> None:
    docs = _bulk_items()
    os_ = _OS({INDEX: docs})

    def another_writer(fake: QueryFakeOpenSearch) -> None:
        # Between the OCC read and the conditional write another writer
        # bumps the doc: the write 409s and must be re-merged, not dropped.
        fake.docs(INDEX)['a0']['updated_at'] = 'later'
        fake._bump(INDEX, 'a0')

    os_.before_bulk = another_writer

    await rbc.cluster_region_residuals(os_)

    assert _boxes(os_, 'a0')['b1'].cluster_id is not None
    assert os_.docs(INDEX)['a0']['updated_at'] != 'later'  # the retried write landed


# ---------------------------------------------------------------------------
# rows + counts
# ---------------------------------------------------------------------------


async def test_count_boxes_counts_boxes_not_items() -> None:
    two = _item('two', [(_box('b1', cluster_id=5), A), (_box('b2', x=0.5, cluster_id=5), B)])
    one = _item('one', [(_box('b1', cluster_id=5), A)])
    other = _item('other', [(_box('b1', cluster_id=6), A)])
    os_ = _OS({INDEX: {'two': two, 'one': one, 'other': other}})

    assert await count_boxes(os_, index=INDEX, states=('accepted',), cluster_id=5) == 3
    assert await count_boxes(os_, index=INDEX, states=('accepted',)) == 4


async def test_rows_match_state_and_cluster_on_the_same_box() -> None:
    # b1 is accepted in cluster 9; b2 is false_positive in cluster 5. Asking
    # for accepted boxes in cluster 5 matches no single box.
    item = _item(
        'x',
        [(_box('b1', cluster_id=9), A), (_box('b2', 'false_positive', x=0.5, cluster_id=5), B)],
    )
    os_ = _OS({INDEX: {'x': item}})

    rows = await scroll_box_rows(os_, index=INDEX, states=('accepted',), cluster_id=5)

    assert rows == []
    rows = await scroll_box_rows(os_, index=INDEX, states=('accepted',), cluster_id=9)
    assert [(r.crop_id, r.box_id) for r in rows] == [('x', 'b1')]


def test_box_filter_selects_state_and_cluster_on_the_same_box() -> None:
    from curation.query_fakes import matches

    item = _item(
        'x',
        [(_box('b1', cluster_id=9), A), (_box('b2', 'false_positive', x=0.5, cluster_id=5), B)],
    )

    # b1 is accepted (cluster 9), b2 is false_positive (cluster 5): no box
    # is both accepted and in cluster 5.
    assert not matches(item, box_filter(('accepted',), cluster_id=5))
    assert matches(item, box_filter(('accepted',), cluster_id=9))
    assert matches(item, box_filter(('false_positive',), cluster_id=5))


async def test_write_box_edits_applies_only_the_named_boxes() -> None:
    item = _item('x', [(_box('b1'), A), (_box('b2', x=0.5), B)])
    os_ = _OS({INDEX: {'x': item}})

    result = await write_box_edits(
        os_,
        index=INDEX,
        edits={'x': {'b2': lambda b: dataclasses.replace(b, cluster_id=3)}},
        respect_human=True,
    )

    assert result['items_written'] == 1
    assert result['boxes_changed'] == 1
    boxes = _boxes(os_, 'x')
    assert boxes['b1'].cluster_id is None
    assert boxes['b2'].cluster_id == 3


async def test_write_box_edits_counts_boxes_and_items_separately() -> None:
    docs = {
        'x': _item('x', [(_box('b1'), A), (_box('b2', x=0.5), B)]),
        'y': _item('y', [(_box('b1'), A)]),
        'z': _item('z', [(_box('b1', cluster_id=3), A)]),
    }
    os_ = _OS({INDEX: docs})

    def to_three(box: RegionBox) -> RegionBox:
        return dataclasses.replace(box, cluster_id=3)

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


async def test_cluster_only_writes_leave_the_revision_an_editor_holds() -> None:
    docs = _bulk_items()
    docs['x'] = _item('x', [(_box('b1'), A), (_box('b2', x=0.5), B)], **{F.revision: 7})
    os_ = _OS({INDEX: docs})

    await rbc.cluster_region_residuals(os_)
    assert os_.docs(INDEX)['x'][F.revision] == 7

    def to_fp(box: RegionBox) -> RegionBox:
        return dataclasses.replace(box, state='false_positive')

    await write_box_edits(os_, index=INDEX, edits={'x': {'b1': to_fp}}, respect_human=True)
    assert os_.docs(INDEX)['x'][F.revision] == 8


async def test_partition_rerun_reports_no_boxes_changed_not_none_assigned() -> None:
    os_ = _OS({INDEX: _bulk_items()})
    first = await rbc.cluster_region_residuals(os_)
    again = await rbc.cluster_region_residuals(os_)

    assert first['n_boxes_changed'] == first['n_boxes'] == 40
    assert again['n_boxes'] == 40
    assert again['n_boxes_changed'] == 0
    assert again['n_items_written'] == 0


# ---------------------------------------------------------------------------
# refine
# ---------------------------------------------------------------------------


def _bucket_items() -> dict[str, dict[str, Any]]:
    docs = {}
    for i in range(3):
        docs[f'a{i}'] = _item(f'a{i}', [(_box('b1', cluster_id=5), _jitter(A, i))])
        docs[f'b{i}'] = _item(f'b{i}', [(_box('b1', cluster_id=5), _jitter(B, i))])
    # An item whose other box lives in a different bucket.
    docs['mix'] = _item(
        'mix', [(_box('b1', cluster_id=5), A), (_box('b2', x=0.5, cluster_id=9), B)]
    )
    return docs


async def test_refine_counts_boxes_when_an_item_has_two_in_the_bucket() -> None:
    docs = _bucket_items()
    docs['pair'] = _item(
        'pair',
        [(_box('b1', cluster_id=5), A), (_box('b2', x=0.5, cluster_id=5), B)],
    )
    os_ = _OS({INDEX: docs})

    result = await rbc.refine_region_cluster(os_, 5)

    assert result['n_boxes'] == 9  # 9 boxes in 8 items
    assert result['n_boxes_updated'] == 9
    assert 'n_members' not in result


async def test_refine_splits_a_bucket_into_sub_clusters_on_the_boxes() -> None:
    os_ = _OS({INDEX: _bucket_items()})

    result = await rbc.refine_region_cluster(os_, 5)

    assert result['action'] == 'refined'
    assert result['n_boxes'] == 7
    assert result['n_subclusters'] == 2
    a = {_boxes(os_, f'a{i}')['b1'].cluster_subid for i in range(3)} | {
        _boxes(os_, 'mix')['b1'].cluster_subid
    }
    b = {_boxes(os_, f'b{i}')['b1'].cluster_subid for i in range(3)}
    assert len(a) == 1
    assert len(b) == 1
    assert a != b
    assert all(sid is not None and sid.startswith('5') for sid in a | b)
    # The sibling box in another bucket is not part of this refine.
    assert _boxes(os_, 'mix')['b2'].cluster_subid is None


async def test_refine_never_stamps_a_box_that_left_the_bucket_during_the_fit() -> None:
    os_ = _OS({INDEX: _bucket_items()})

    def a0_moves_bucket(fake: QueryFakeOpenSearch) -> None:
        doc = fake.docs(INDEX)['a0']
        moved = dataclasses.replace(read_boxes(doc, F)[0], cluster_id=99)
        doc.update(boxes_write_fields([moved], current_src=doc))
        fake._bump(INDEX, 'a0')

    os_.after_scan = a0_moves_bucket

    await rbc.refine_region_cluster(os_, 5)

    moved = _boxes(os_, 'a0')['b1']
    assert moved.cluster_id == 99
    assert moved.cluster_subid is None
    assert _boxes(os_, 'a1')['b1'].cluster_subid is not None


async def test_refine_too_small_a_bucket_writes_nothing() -> None:
    docs = {'a': _item('a', [(_box('b1', cluster_id=5), A)])}
    os_ = _OS({INDEX: docs})
    before = copy.deepcopy(os_.docs(INDEX))

    result = await rbc.refine_region_cluster(os_, 5)

    assert result['action'] == 'skipped_too_small'
    assert os_.docs(INDEX) == before


# ---------------------------------------------------------------------------
# FP centroids + auto pull
# ---------------------------------------------------------------------------


@pytest.fixture
def fp_store_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr('src.services.detection.fp_store.fp_store_dir', lambda *_a, **_k: tmp_path)
    return tmp_path


async def test_fp_centroids_are_sub_typed_from_fp_boxes_only(fp_store_dir: Any) -> None:
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    fp1 = _box('b1', 'false_positive', cluster_id=FP, cluster_distance=0.0)
    fp2 = _box('b2', 'false_positive', x=0.5, cluster_id=FP, cluster_distance=0.0)
    docs = {
        # a good box (vector along A) beside an FP box (vector along B)
        'x': _item('x', [(_box('b1'), A), (dataclasses.replace(fp2), B)]),
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
    x, y = _boxes(os_, 'x'), _boxes(os_, 'y')
    assert x['b2'].cluster_subid == f'{FP}a'
    assert y['b1'].cluster_subid == f'{FP}a'
    assert x['b1'].cluster_subid is None  # the good box of the same item
    assert x['b1'].cluster_id is None


async def test_fp_centroids_without_fp_boxes_is_a_noop(fp_store_dir: Any) -> None:
    os_ = _OS({INDEX: {'z': _item('z', [(_box('b1'), A)])}})

    result = await rbc.build_region_fp_centroids(os_)

    assert result == {'status': 'skipped', 'reason': 'no_fp_embeddings', 'n_boxes': 0}


def _save_centroid(fp_store_dir: Any, vector: tuple[float, float, float]) -> None:
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    FalsePositiveCentroidStore().save(
        np.asarray([_vec(*vector)], dtype=np.float32),
        {'trained_at': 't', 'k': 1, 'n_boxes': 1, 'subids': [f'{FP}a'], 'dim': 3},
    )


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
    assert (box.cluster_id, box.cluster_subid) == (FP, f'{FP}a')
    assert box.cluster_distance == pytest.approx(0.0, abs=1e-5)
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
    assert boxes['b1'].cluster_id is None


async def test_auto_pull_without_centroids_is_a_noop(fp_store_dir: Any) -> None:
    os_ = _OS({INDEX: {'a': _item('a', [(_box('b1'), B)])}})

    result = await rbc.auto_assign_fp_from_centroids(os_)

    assert result['status'] == 'skipped'
    assert _boxes(os_, 'a')['b1'].state == 'accepted'


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
