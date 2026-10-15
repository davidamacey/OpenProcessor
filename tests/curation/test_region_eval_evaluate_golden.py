"""Characterization of :func:`src.services.curation.region_eval.evaluate`.

A ten-frame cohort drives every branch of the scorer (not ingested, pending,
no items, backgrounds with and without detections, near-duplicate merging,
every miss reason, a score-less prediction, a split-less frame) and the
complete ``EvalResult`` is compared with a golden file written from the code
before the function was decomposed. Regenerate deliberately with
``OP_REGEN_REGION_EVAL_GOLDEN=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from src.services.curation.region_eval import evaluate
from src.services.curation.region_eval_data import CohortImage, RegionRecord


GOLDEN = Path(__file__).parent / 'golden' / 'region_eval_evaluate.json'

B1 = (0.10, 0.10, 0.40, 0.40)
B2 = (0.60, 0.60, 0.90, 0.90)
B3 = (0.20, 0.20, 0.50, 0.50)


def _rec(cid: str, status: str, box: Any, det: str = 'yolo', score: float | None = 0.9):
    return RegionRecord(crop_id=cid, status=status, detector=det, score=score, box=box)


def _cohort() -> tuple[list[CohortImage], dict[str, list[RegionRecord]]]:
    def img(key: str, gt: list[Any], split: str, image_id: str | None) -> CohortImage:
        return CohortImage(
            key=f'{key}.jpg', server_path=f'/s/{key}.jpg', gt=gt, split=split, image_id=image_id
        )

    cohort = [
        img('a', [B1, B2], 'train', 'A'),
        img('b', [B3], 'train', 'B'),
        img('c', [B3], 'train', None),
        img('d', [B1], 'val', 'D'),
        img('e', [], 'val', 'E'),
        img('f', [], 'val', 'F'),
        img('g', [B1], 'train', 'G'),
        img('h', [B2], '', 'H'),
        img('i', [B3], 'train', 'I'),
        img('j', [B2], 'train', 'J'),
    ]
    regions = {
        'A': [
            _rec('a1', 'detected', (0.11, 0.11, 0.41, 0.41), score=0.95),
            _rec('a2', 'detected', (0.11, 0.11, 0.40, 0.41), det='sam', score=0.80),
            _rec('a3', 'detected', (0.70, 0.70, 0.99, 0.99), score=0.7),
            _rec('a4', 'verify_rejected', (0.60, 0.60, 0.89, 0.90), det='sam', score=0.6),
        ],
        'D': [_rec('d1', 'pending_verification', (0.1, 0.1, 0.4, 0.4))],
        'E': [_rec('e1', 'detected', (0.3, 0.3, 0.6, 0.6), score=0.5)],
        'F': [
            _rec('f1', 'false_positive', (0.3, 0.3, 0.6, 0.6)),
            _rec('f2', 'no_region_visible', None, score=None),
        ],
        'G': [_rec('g1', 'detected', (0.7, 0.7, 0.95, 0.95))],
        'H': [_rec('h1', 'detected', (0.6, 0.6, 0.9, 0.9), det='sam')],
        'I': [_rec('i1', 'detection_failed', None, score=None)],
        'J': [
            _rec('j1', 'detected', (0.55, 0.55, 0.8, 0.8), score=None),
            _rec('j2', 'detected', (0.0, 0.0, 0.2, 0.2), score=0.3),
        ],
    }
    return cohort, regions


CASES: dict[str, dict[str, Any]] = {
    'defaults': {},
    'loose_no_dedup': {
        'iou_threshold': 0.3,
        'dedup_iou': 0.0,
        'accepted': ('detected', 'verify_rejected'),
    },
}


def _snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, kwargs in CASES.items():
        cohort, regions = _cohort()
        res = evaluate(cohort, regions, **kwargs)
        out[name] = {
            'summary': res.summary,
            'misses': res.misses,
            'false_positives': res.false_positives,
            'pending_images': res.pending_images,
        }
    return json.loads(json.dumps(out))


def test_evaluate_matches_the_golden_result() -> None:
    snap = _snapshot()
    if os.environ.get('OP_REGEN_REGION_EVAL_GOLDEN') == '1':
        GOLDEN.write_text(json.dumps(snap, indent=2, sort_keys=True) + '\n')
        pytest.skip('golden regenerated')
    assert snap == json.loads(GOLDEN.read_text())


def test_golden_cohort_reaches_every_miss_reason_and_state() -> None:
    cohort, regions = _cohort()
    res = evaluate(cohort, regions, accepted=('detected',))
    assert set(res.summary['miss_reasons']) == {
        'no_items',
        'no_region',
        'low_iou',
        'status:verify_rejected',
    }
    total = res.summary['total']
    assert (total['not_ingested'], total['pending_images'], total['no_item_images']) == (1, 1, 1)
    assert total['duplicates_merged'] == 1
