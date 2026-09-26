"""An item whose source image can't be read is a failure, never "no region".

A missing mount once made every crop unreadable and the worker recorded all
of them as ``no_region_box`` -- a terminal "looked and found nothing" verdict
that silently corrupted the dataset. Unreadable input must land in the
retryable ``detection_failed`` state with a reason, and the worker must never
map a missing crop to ``no_region_box``.
"""

from __future__ import annotations

import re
from pathlib import Path

from scripts.curation.worker.state import _ItemTask, unreadable_crop_update
from src.config import get_region_fields
from src.config.region_state import RegionStatus


def _task() -> _ItemTask:
    return _ItemTask(
        crop_id='c1',
        image_path='/missing/frame.jpg',
        item_bbox_norm=(0.1, 0.1, 0.5, 0.5),
        region_status='pending_detection',
        class_name='car',
        request_id='r1',
    )


def test_unreadable_crop_is_detection_failed_with_reason() -> None:
    F = get_region_fields()
    t = _task()
    t.detection_trace.append('earlier:step')
    update = unreadable_crop_update(t)
    assert update[F.status] == RegionStatus.DETECTION_FAILED
    assert update[F.reason] == 'image_unavailable'
    assert update[F.detector_chain] == ['earlier:step', 'worker:image_unavailable']


def test_runner_never_maps_a_missing_crop_to_no_region_box() -> None:
    src = Path('scripts/curation/worker/runner.py').read_text()
    # Every branch that handles a missing crop must route through the helper.
    for m in re.finditer(r'(t\.crop_jpeg is None|i in bad_indices):\n((?:[ \t]+.*\n){1,4})', src):
        body = m.group(2)
        assert 'NO_REGION_BOX' not in body, f'missing crop mapped to no_region_box:\n{m.group(0)}'
    # The four stages that can hold an unreadable crop all use the helper.
    assert src.count('t.update_doc = unreadable_crop_update(t)') == 4
