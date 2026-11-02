"""The bench-only item replicator (scripts/bench/replicate_items.py)."""

from __future__ import annotations

import numpy as np
import orjson

from scripts.bench import replicate_items as rep


def test_copy_doc_keeps_fields_renames_and_renormalises() -> None:
    src = {'pe_embedding': [1.0, 0.0, 0.0, 0.0], 'detector_class_name': 'car', 'image_id': 'i1'}
    noise = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    doc_id, copy = rep.copy_doc('c1', src, 3, noise, sigma=0.5)
    assert doc_id == 'c1~r3'
    assert copy['detector_class_name'] == 'car'
    assert copy['image_id'] == 'i1'
    vec = np.asarray(copy['pe_embedding'])
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6
    assert vec[1] > 0  # jittered, not an exact duplicate
    assert src['pe_embedding'] == [1.0, 0.0, 0.0, 0.0]  # the original is untouched


def test_bulk_body_is_action_then_source_lines() -> None:
    body = rep.bulk_body([('a', {'x': 1}), ('b', {'x': 2})])
    lines = [orjson.loads(line) for line in body.strip().split(b'\n')]
    assert lines == [{'index': {'_id': 'a'}}, {'x': 1}, {'index': {'_id': 'b'}}, {'x': 2}]
    assert body.endswith(b'\n')
