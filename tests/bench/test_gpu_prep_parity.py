"""``scripts/bench/gpu_prep_parity.py``: the comparison that gates a change to ``gpu_image_prep``."""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.bench import gpu_prep_parity as par


def _row(scale: float = 1.0, jpeg: bytes = b'\xff\xd8a') -> dict[str, Any]:
    return {
        'status': np.array([b'ok'], dtype=object),
        'size': np.array([640, 480], np.int32),
        'letterbox': np.array([1.0, 0.0, 80.0]),
        'det_boxes': np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.9, 0.9]], np.float32),
        'det_scores': np.array([0.5, 0.9], np.float32),
        'det_classes': np.array([1, 2], np.int32),
        'whole_embedding': np.array([1.0, 2.0, 3.0], np.float32) * scale,
        'crop_embeddings': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32),
        'blur_sums': np.array([[100, 1, 2], [10, 1, 2], [20, 3, 4]], np.int64),
        'crop_jpegs': np.array([jpeg, jpeg], dtype=object),
    }


def test_identical_answers_are_byte_equal() -> None:
    r = par.compare_pair(_row(), _row())
    assert r['bytes_equal'] is True
    assert r['whole_cos'] == 1.0


def test_detections_are_paired_by_score_not_by_position() -> None:
    a, b = _row(), _row()
    for k in ('det_boxes', 'det_scores', 'det_classes', 'crop_embeddings'):
        b[k] = b[k][::-1]
    b['blur_sums'] = np.array([[100, 1, 2], [20, 3, 4], [10, 1, 2]], np.int64)
    assert par.compare_pair(a, b)['bytes_equal'] is True


def test_a_changed_crop_jpeg_is_reported() -> None:
    r = par.compare_pair(_row(), _row(jpeg=b'\xff\xd8b'))
    assert r['crop_jpeg_equal'] is False
    assert r['bytes_equal'] is False


def test_a_different_status_is_reported_and_stops_the_comparison() -> None:
    b = _row()
    b['status'] = np.array([b'gpu_memory'], dtype=object)
    r = par.compare_pair(_row(), b)
    assert r == {'status_equal': False}


def test_a_moved_box_is_measured() -> None:
    b = _row()
    b['det_boxes'] = b['det_boxes'] + np.float32(0.01)
    assert abs(par.compare_pair(_row(), b)['box_max_abs'] - 0.01) < 1e-6


def test_summary_counts_the_differences() -> None:
    rows = [par.compare_pair(_row(), _row()), par.compare_pair(_row(), _row(jpeg=b'x'))]
    s = par.summarize(rows)
    assert (s['images'], s['bytes_equal'], s['jpeg_differs']) == (2, 1, 1)


def test_capture_round_trips_through_a_file(tmp_path: Any) -> None:
    from pathlib import Path

    path = tmp_path / 'c.npz'
    par.save(path, [Path('a.jpg'), Path('b.jpg')], [_row(), _row(2.0)])
    back = par.load(path)
    assert len(back) == 2
    assert par.compare_pair(back[0], _row())['bytes_equal'] is True
    assert float(par.compare_pair(back[1], _row(2.0))['whole_cos']) == 1.0
