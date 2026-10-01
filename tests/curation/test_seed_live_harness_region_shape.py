"""The live-harness seed must write region vectors the clustering readers accept."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from scripts.curation.seed_live_harness import _region_fields
from src.config import get_region_fields
from src.services.curation.region_box_embeddings import join_box_vectors
from src.services.curation.region_boxes import read_boxes


F = get_region_fields()


@pytest.mark.parametrize('region', ['detected', 'false_positive'])
@pytest.mark.parametrize('global_index', [0, 1, 2])
def test_every_seeded_box_has_a_current_vector(region: str, global_index: int) -> None:
    vec = np.random.default_rng(0).normal(size=16)
    vec /= np.linalg.norm(vec)
    doc = _region_fields(
        F, region, global_index=global_index, local_idx=0, vector=vec, now=datetime.now(UTC)
    )
    joined = join_box_vectors(doc, F)
    assert [b.box_id for b, _ in joined] == [b.box_id for b in read_boxes(doc, F)]
