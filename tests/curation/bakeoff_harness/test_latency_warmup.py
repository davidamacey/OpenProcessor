"""A holdout no larger than ``--warmup`` must not silently report zero latency."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from curation.bakeoff_harness._helpers import _img
from scripts.curation.bakeoff import metrics
from scripts.curation.bakeoff.run import _run_inference


class _Backend:
    def detect(self, _frame: object) -> list[object]:
        return []


def _ds(tmp_path, n: int, *, readable: bool = True):
    images = []
    for i in range(n):
        path = tmp_path / f'{i}.png'
        if readable:
            _img(path)
        images.append(SimpleNamespace(image_id=i, path=path))
    return SimpleNamespace(images=images)


@pytest.mark.parametrize('n', [1, 3])
def test_small_holdout_keeps_a_timed_frame_and_records_warmup(tmp_path, n):
    _, latencies, warmup_used = _run_inference(_Backend(), _ds(tmp_path, n), warmup=3)
    assert len(latencies) >= 1
    assert warmup_used == n - 1


def test_large_holdout_uses_requested_warmup(tmp_path):
    _, latencies, warmup_used = _run_inference(_Backend(), _ds(tmp_path, 6), warmup=3)
    assert warmup_used == 3
    assert len(latencies) == 3


def test_no_timed_frame_is_refused_not_zeroed(tmp_path):
    with pytest.raises(SystemExit, match='latency'):
        _run_inference(_Backend(), _ds(tmp_path, 2, readable=False), warmup=3)


def test_percentiles_refuses_empty_sample():
    with pytest.raises(ValueError, match='empty'):
        metrics.percentiles([])
