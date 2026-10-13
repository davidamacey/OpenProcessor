"""Tests for the pure parts of scripts/bench/os_write_ceiling.py."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import orjson
import pytest

from scripts.bench import os_write_ceiling as o


MAPPING: dict[str, Any] = {
    'properties': {
        'pe_embedding': {
            'type': 'knn_vector',
            'dimension': 4,
            'method': {'name': 'hnsw', 'parameters': {'ef_construction': 512, 'm': 16}},
        },
        'region_box_embeddings': {
            'type': 'nested',
            'properties': {
                'embedding': {
                    'type': 'knn_vector',
                    'dimension': 4,
                    'method': {'name': 'hnsw', 'parameters': {'ef_construction': 512, 'm': 16}},
                }
            },
        },
        'label': {'type': 'keyword'},
    }
}


def test_with_ef_construction_sets_every_vector_and_copies() -> None:
    out = o.with_ef_construction(MAPPING, 128)
    assert out['properties']['pe_embedding']['method']['parameters'] == {
        'ef_construction': 128,
        'm': 16,
    }
    nested = out['properties']['region_box_embeddings']['properties']['embedding']
    assert nested['method']['parameters']['ef_construction'] == 128
    assert MAPPING['properties']['pe_embedding']['method']['parameters']['ef_construction'] == 512
    assert o.with_ef_construction(MAPPING, None) == MAPPING


def test_index_body_keeps_shipped_settings_and_sets_refresh_only_when_asked() -> None:
    source: dict[str, Any] = {
        'settings': {
            'index': {
                'number_of_shards': '1',
                'number_of_replicas': '0',
                'knn': 'true',
                'uuid': 'drop-me',
                'creation_date': 'drop-me',
            }
        },
        'mappings': MAPPING,
    }
    shipped = o.index_body(source, 'shipped', None)['settings']['index']
    assert shipped == {'number_of_shards': '1', 'number_of_replicas': '0', 'knn': 'true'}
    assert o.index_body(source, '-1', None)['settings']['index']['refresh_interval'] == '-1'


def test_make_jobs_covers_every_document_once_and_interleaves() -> None:
    jobs = o.make_jobs(n_items=2500, n_images=400, bulk=1000)
    for kind, total in (('items', 2500), ('images', 400)):
        spans = sorted((a, b) for k, a, b in jobs if k == kind)
        assert spans[0][0] == 0
        assert spans[-1][1] == total
        assert all(x[1] == y[0] for x, y in itertools.pairwise(spans))
    assert [k for k, _, _ in jobs[:3]] != ['items'] * 3
    assert o.make_jobs(0, 5, 10) == [('images', 0, 5)]


def test_synth_vectors_are_unit_length_and_follow_the_pool() -> None:
    rng = np.random.default_rng(1)
    pool = o.unit_rows(rng.standard_normal((5, 64), dtype=np.float32))
    near = o.synth_vectors(pool, 200, 64, rng)
    assert near.shape == (200, 64)
    assert np.allclose(np.linalg.norm(near, axis=1), 1.0, atol=1e-5)
    assert (near @ pool.T).max(axis=1).min() > 0.8  # each stays close to some pool vector
    iid = o.synth_vectors(None, 200, 64, rng)
    assert (iid @ pool.T).max(axis=1).mean() < 0.6


def test_zero_noise_returns_distinct_real_rows_and_refuses_a_short_pool() -> None:
    rng = np.random.default_rng(2)
    pool = o.unit_rows(rng.standard_normal((50, 16), dtype=np.float32))
    real = o.synth_vectors(pool, 30, 16, rng, noise=0.0)
    assert len({tuple(r) for r in real.round(6)}) == 30
    assert all(np.isclose(pool @ r, 1.0, atol=1e-5).any() for r in real)
    with pytest.raises(ValueError, match='50'):
        o.synth_vectors(pool, 51, 16, rng, noise=0.0)


def test_bulk_body_is_valid_ndjson_with_unique_ids() -> None:
    docs = [orjson.dumps({'n': i}) for i in range(5)]
    body = o.bulk_body('w0os-x', 'items', docs, 1, 4)
    lines = body.decode().splitlines()
    assert len(lines) == 6
    actions = [orjson.loads(ln)['index'] for ln in lines[0::2]]
    assert [a['_id'] for a in actions] == ['i1', 'i2', 'i3']
    assert {a['_index'] for a in actions} == {'w0os-x'}
    assert [orjson.loads(ln)['n'] for ln in lines[1::2]] == [1, 2, 3]


def test_arm_config_overrides_the_shipped_arm() -> None:
    cfg = o.arm_config('ef128')
    assert cfg['ef'] == 128
    assert cfg['bulk'] == 1000
    assert o.arm_config('shipped')['ef'] is None
    with pytest.raises(KeyError):
        o.arm_config('nope')
