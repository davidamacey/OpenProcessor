"""The live-harness seed must write region vectors the clustering readers accept."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from scripts.curation.seed_live_harness import _region_fields
from src.config import get_region_fields
from src.domain.region_boxes import read_boxes
from src.services.curation.region_box_embeddings import join_box_vectors


F = get_region_fields()


@pytest.mark.parametrize('region', ['detected', 'false_positive'])
@pytest.mark.parametrize('global_index', [0, 1, 2])
def test_every_seeded_box_has_a_current_vector(region: str, global_index: int) -> None:
    vec = np.random.default_rng(0).normal(size=16)
    vec /= np.linalg.norm(vec)
    doc = _region_fields(F, region, global_index=global_index, vector=vec, now=datetime.now(UTC))
    joined = join_box_vectors(doc, F)
    assert [b.box_id for b, _ in joined] == [b.box_id for b in read_boxes(doc, F)]


def test_seeded_items_carry_the_registry_id_of_their_class_name() -> None:
    """A registry that already holds the region class hands the shipping classes ids from 1, so an
    item's class_id is the id the registry gave its name, never the name's position (class
    identity is the name)."""
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from scripts.curation.seed_live_harness import CLASS_NAMES, _build_items, _write_images

    registry_ids = {name: 100 + i for i, name in enumerate(CLASS_NAMES)}
    cfg = SimpleNamespace(encoder_embedding_dim=8)
    with tempfile.TemporaryDirectory() as root:
        images = _write_images(Path(root), 4)
        docs = _build_items(cfg, F, images, registry_ids)
    labelled = [d for d in docs if d['class_name'] is not None]
    assert labelled
    assert all(d['class_id'] == registry_ids[d['class_name']] for d in labelled)
