"""A re-encoded copy of a photo (same pixels, tiny vector noise) lands in one dedup group with
its original at the default 0.98 cut, through the same exact search, cut and grouping a full
build runs; unrelated photos stay alone. Vectors are synthetic: this pins the grouping logic,
not an encoder's behaviour (the live check compares real vectors)."""

from __future__ import annotations

import numpy as np

from src.domain import near_dup as rules
from src.services.near_duplicates.candidates import exact_edges, unit_rows


def test_copies_group_with_their_originals_and_strangers_do_not() -> None:
    rng = np.random.default_rng(7)
    originals = rng.normal(size=(40, 256)).astype(np.float32)
    copies = originals[:3] + rng.normal(scale=0.02, size=(3, 256)).astype(np.float32)
    mat = unit_rows(np.vstack([originals, copies]))
    ids = [f'img{n:03d}' for n in range(len(mat))]

    edges = exact_edges(ids, mat, k=64, min_cosine=rules.DEFAULT_IMAGE_DEDUP_COSINE)
    groups = rules.group_members(
        rules.effective_edges(edges, rules.DEFAULT_IMAGE_DEDUP_COSINE), rules.DEFAULT_MAX_GROUP_SIZE
    ).groups

    assert groups == [['img000', 'img040'], ['img001', 'img041'], ['img002', 'img042']]
