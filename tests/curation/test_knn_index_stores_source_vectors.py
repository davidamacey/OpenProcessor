"""Found live on OpenSearch 3.6: with derived source (the default for a kNN
index) a nested ``knn_vector`` read through a ``_source`` include came back
as the number ``1``, so every per-box embedding read (box clustering, FP
centroids, refine) failed. Every index holding vectors must keep them in
``_source``."""

from __future__ import annotations

import pytest

from src.clients import curation_opensearch as cop
from src.config import IndexRole


@pytest.mark.parametrize('role', [IndexRole.ITEMS, IndexRole.IMAGES])
def test_vector_indexes_do_not_derive_source(role: IndexRole) -> None:
    index = cop.INDEX_BODIES[role]['settings']['index']
    assert index['knn'] is True
    assert index['knn.derived_source.enabled'] is False
