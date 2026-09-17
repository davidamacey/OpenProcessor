"""Tests for :mod:`src.services.detection.frame_dedup`.

No direct coverage exists on the reference branch for this module —
written fresh. Covers the pure-numpy union-find grouping/collapse
logic synchronously, plus the async ``dedup_rows_by_embedding``
orchestration against a fake OpenSearch client (no live stack).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.config import CurationConfig
from src.services.detection.frame_dedup import (
    DEFAULT_FRAME_DEDUP_THRESHOLD,
    collapse_keep_indices,
    dedup_rows_by_embedding,
    most_central,
    near_dup_groups,
)


def _unit(vecs: list[list[float]]) -> np.ndarray:
    mat = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / norms).astype(np.float32)


class TestNearDupGroups:
    def test_fewer_than_two_rows_returns_empty(self) -> None:
        assert near_dup_groups(_unit([[1.0, 0.0]])) == []

    def test_identical_vectors_group_together(self) -> None:
        mat = _unit([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        groups = near_dup_groups(mat, threshold=0.98)
        assert groups == [[0, 1]]

    def test_orthogonal_vectors_do_not_group(self) -> None:
        mat = _unit([[1.0, 0.0], [0.0, 1.0]])
        assert near_dup_groups(mat, threshold=0.98) == []

    def test_transitive_grouping_via_chain(self) -> None:
        # a~b (cos ~0.999), b~c (cos ~0.999), a and c both above threshold
        # transitively through b.
        mat = _unit([[1.0, 0.0], [0.999, 0.0447], [0.996, 0.0894]])
        groups = near_dup_groups(mat, threshold=0.98)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}


class TestMostCentral:
    def test_returns_index_of_most_representative_member(self) -> None:
        # Third vector is the average direction of the first two -> most central.
        mat = _unit([[1.0, 0.1], [0.1, 1.0], [1.0, 1.0]])
        assert most_central(mat, [0, 1, 2]) == 2


class TestCollapseKeepIndices:
    def test_all_singletons_are_kept(self) -> None:
        mat = _unit([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        keep, n_groups, n_dropped = collapse_keep_indices(mat, threshold=0.98)
        assert keep == [0, 1, 2]
        assert n_groups == 0
        assert n_dropped == 0

    def test_duplicate_group_collapses_to_one_survivor(self) -> None:
        mat = _unit([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        keep, n_groups, n_dropped = collapse_keep_indices(mat, threshold=0.98)
        assert n_groups == 1
        assert n_dropped == 1
        assert 2 in keep
        assert len(keep) == 2

    def test_prefer_flag_wins_the_survivor_slot(self) -> None:
        mat = _unit([[1.0, 0.0], [1.0, 0.0]])
        prefer = np.array([False, True])
        keep, _, _ = collapse_keep_indices(mat, threshold=0.98, prefer=prefer)
        assert keep == [1]


@dataclass
class _Row:
    image_id: str
    has_test_crop: bool
    tag: str = ''


class _FakeOpenSearch:
    """Minimal async fake exposing only the ``mget`` shape frame_dedup needs."""

    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self._embeddings = embeddings

    async def mget(self, index: str, body: dict, _source: list[str]) -> dict:  # noqa: ARG002
        docs = []
        for image_id in body['ids']:
            vec = self._embeddings.get(image_id)
            source = {'image_id': image_id}
            if vec is not None:
                source['pe_embedding'] = vec
            docs.append({'_id': image_id, '_source': source})
        return {'docs': docs}


class TestDedupRowsByEmbedding:
    @pytest.mark.asyncio
    async def test_defaults_index_to_configured_images_index(self) -> None:
        seen_index: list[str] = []

        class _RecordingOpenSearch(_FakeOpenSearch):
            async def mget(self, index: str, body: dict, _source: list[str]) -> dict:
                seen_index.append(index)
                return await super().mget(index, body, _source)

        client = _RecordingOpenSearch({'a': [1.0, 0.0]})
        rows = [_Row(image_id='a', has_test_crop=False)]
        cfg = CurationConfig(images_index='custom_images_idx')
        await dedup_rows_by_embedding(client, rows, config=cfg)
        assert seen_index == ['custom_images_idx']

    @pytest.mark.asyncio
    async def test_rows_without_image_id_are_kept_untouched(self) -> None:
        client = _FakeOpenSearch({})
        rows = [_Row(image_id='', has_test_crop=False)]
        kept, stats = await dedup_rows_by_embedding(client, rows)
        assert kept == rows
        assert stats['n_frames'] == 0

    @pytest.mark.asyncio
    async def test_rows_with_missing_embedding_are_kept(self) -> None:
        client = _FakeOpenSearch({})
        rows = [_Row(image_id='no-embed', has_test_crop=False)]
        kept, stats = await dedup_rows_by_embedding(client, rows)
        assert kept == rows
        assert stats['n_frames_missing_embedding'] == 1

    @pytest.mark.asyncio
    async def test_near_dup_frames_collapse_all_their_rows(self) -> None:
        # Two frames (f1, f2) are near-dup; f1 has two crop rows, f2 has one.
        # Collapsing must keep or drop ALL rows of a frame together.
        embeddings = {
            'f1': [1.0, 0.0],
            'f2': [1.0, 0.0],
            'f3': [0.0, 1.0],
        }
        client = _FakeOpenSearch(embeddings)
        rows = [
            _Row(image_id='f1', has_test_crop=False, tag='f1-a'),
            _Row(image_id='f1', has_test_crop=False, tag='f1-b'),
            _Row(image_id='f2', has_test_crop=False, tag='f2-a'),
            _Row(image_id='f3', has_test_crop=False, tag='f3-a'),
        ]
        kept, stats = await dedup_rows_by_embedding(client, rows, threshold=0.98)
        kept_frames = {r.image_id for r in kept}
        # Exactly one of f1/f2 survives, and f3 always survives (distinct).
        assert 'f3' in kept_frames
        assert len(kept_frames) == 2
        assert stats['near_dup_groups'] == 1
        assert stats['frames_dropped'] == 1
        # Whichever frame survived, ALL of its rows must be present together.
        surviving = kept_frames - {'f3'}
        assert surviving in ({'f1'}, {'f2'})
        if surviving == {'f1'}:
            assert len([r for r in kept if r.image_id == 'f1']) == 2
        else:
            assert len([r for r in kept if r.image_id == 'f2']) == 1

    @pytest.mark.asyncio
    async def test_held_out_frame_is_preferred_as_survivor(self) -> None:
        embeddings = {'f1': [1.0, 0.0], 'f2': [1.0, 0.0]}
        client = _FakeOpenSearch(embeddings)
        rows = [
            _Row(image_id='f1', has_test_crop=False),
            _Row(image_id='f2', has_test_crop=True),  # held-out -> must survive
        ]
        kept, _ = await dedup_rows_by_embedding(client, rows, threshold=0.98)
        assert {r.image_id for r in kept} == {'f2'}

    @pytest.mark.asyncio
    async def test_default_threshold_is_the_documented_constant(self) -> None:
        assert DEFAULT_FRAME_DEDUP_THRESHOLD == 0.98
