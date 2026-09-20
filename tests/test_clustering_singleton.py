"""Coverage for the ``ClusteringService`` singleton bootstrap and its
``load_index``/``load_all_indexes`` fail-open branch (plan Wave 5 T-6).

This module is pure collateral test-drop damage — it predates the
curation port and has zero import dependency on it — so this is a
straight restoration, not a genericization port.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

import src.services.clustering as clustering_mod
from src.services.clustering import ClusterIndex, ClusteringService, get_clustering_service


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _reset_singleton() -> Iterator[None]:
    """The module-level singleton survives across tests in the same
    session; clear it before and after so each test exercises a clean
    first-creation branch."""
    clustering_mod._clustering_service = None
    yield
    clustering_mod._clustering_service = None


def test_get_clustering_service_constructs_with_given_args(tmp_path: Path) -> None:
    svc = get_clustering_service(index_dir=tmp_path, use_gpu=False)
    assert isinstance(svc, ClusteringService)
    assert svc.index_dir == tmp_path
    assert svc.use_gpu is False


def test_get_clustering_service_is_a_singleton(tmp_path: Path) -> None:
    """Subsequent calls return the SAME instance and ignore new args —
    a literal expectation, not a re-derived one: a second call with a
    different index_dir must NOT rebuild the service."""
    a = get_clustering_service(index_dir=tmp_path, use_gpu=False)
    b = get_clustering_service(index_dir=tmp_path / 'other', use_gpu=True)
    assert a is b
    assert b.index_dir == tmp_path  # unchanged — the first call won


def test_load_all_indexes_calls_load_index_for_every_cluster_index(tmp_path: Path) -> None:
    svc = ClusteringService(index_dir=tmp_path, use_gpu=False)
    fake_load = MagicMock(return_value=False)
    svc.load_index = fake_load  # type: ignore[method-assign]

    results = svc.load_all_indexes()

    assert set(results.keys()) == set(ClusterIndex)
    assert fake_load.call_count == len(ClusterIndex)
    for index_name in ClusterIndex:
        fake_load.assert_any_call(index_name)


def test_load_index_returns_false_when_no_file_on_disk(tmp_path: Path) -> None:
    svc = ClusteringService(index_dir=tmp_path, use_gpu=False)
    assert svc.load_index(ClusterIndex.GLOBAL) is False
    assert ClusterIndex.GLOBAL not in svc._indexes


def test_load_index_fail_open_swallows_a_corrupt_index_file(tmp_path: Path) -> None:
    """The fail-open branch: a present-but-unreadable index file must
    not raise — ``load_index`` logs and returns False so a corrupt
    on-disk index degrades to "untrained", not a service outage."""
    index_path = tmp_path / f'{ClusterIndex.GLOBAL.value}.index'
    index_path.write_bytes(b'not a real faiss index')

    svc = ClusteringService(index_dir=tmp_path, use_gpu=False)
    assert svc.load_index(ClusterIndex.GLOBAL) is False
    assert ClusterIndex.GLOBAL not in svc._indexes


def test_load_index_success_populates_indexes_and_metadata(tmp_path: Path) -> None:
    """Non-regression: a real index file still loads successfully and
    populates both ``_indexes`` and ``_training_metadata`` when present."""
    svc = ClusteringService(index_dir=tmp_path, use_gpu=False, embedding_dim=8)
    faiss = pytest.importorskip('faiss')
    quantizer = faiss.IndexFlatL2(8)
    index = faiss.IndexIVFFlat(quantizer, 8, 4)
    import numpy as np

    training = np.random.default_rng(0).random((64, 8), dtype='float32')
    index.train(training)

    index_path = tmp_path / f'{ClusterIndex.GLOBAL.value}.index'
    faiss.write_index(index, str(index_path))
    meta_path = tmp_path / f'{ClusterIndex.GLOBAL.value}.meta.npy'
    np.save(meta_path, {'trained_at': '2026-01-01T00:00:00'})

    assert svc.load_index(ClusterIndex.GLOBAL) is True
    assert ClusterIndex.GLOBAL in svc._indexes
    assert svc._training_metadata[ClusterIndex.GLOBAL]['trained_at'] == '2026-01-01T00:00:00'
