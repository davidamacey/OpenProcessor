"""Tests for the region-cascade-vs-ground-truth evaluator
(:mod:`src.services.curation.region_eval` and
``scripts/curation/eval_regions_vs_gt.py``).

Index reads go through :class:`curation.query_fakes.QueryFakeOpenSearch`, which
evaluates the ``terms`` selections for real, so the cohort -> image_id ->
items join is exercised, not stubbed.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.config import CurationConfig, RegionStatus
from src.config.region_fields import RegionFields, get_region_fields
from src.services.curation.region_eval import (
    NO_DETECTOR,
    CohortImage,
    RegionFrameError,
    evaluate,
    greedy_match,
    iou,
    parse_yolo_labels,
    region_records,
    run_eval,
    to_source_frame,
    yolo_to_xyxy,
)
from src.services.detection.cascade_detect.sanity import crop_norm_to_source_norm


if TYPE_CHECKING:
    from pathlib import Path


IMAGES = 'test_images'
ITEMS = 'test_items'
CFG = CurationConfig(images_index=IMAGES, items_index=ITEMS)
F = get_region_fields()
DET = RegionStatus.DETECTED.value


# =============================================================================
# Geometry / parsing
# =============================================================================


def test_yolo_to_xyxy_and_clamp() -> None:
    assert yolo_to_xyxy(0.5, 0.5, 0.2, 0.4) == pytest.approx((0.4, 0.3, 0.6, 0.7))
    assert yolo_to_xyxy(0.05, 0.5, 0.2, 0.2) == pytest.approx((0.0, 0.4, 0.15, 0.6))


def test_parse_yolo_labels_rows_segments_and_class_filter() -> None:
    text = (
        '# comment\n'
        '0 0.5 0.5 0.2 0.2\n'
        '\n'
        '1 0.1 0.1 0.1 0.1\n'
        '0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n'  # polygon -> its bounding box
        'garbage row\n'
        '0 0.5 0.5\n'
    )
    boxes = parse_yolo_labels(text)
    assert len(boxes) == 3
    assert boxes[2] == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert parse_yolo_labels(text, [1]) == [pytest.approx((0.05, 0.05, 0.15, 0.15))]
    assert parse_yolo_labels('') == []


def test_iou_basic() -> None:
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == pytest.approx(1.0)
    assert iou((0, 0, 0.5, 1), (0.5, 0, 1, 1)) == 0.0
    assert iou((0, 0, 0.5, 0.5), (0.25, 0.25, 0.75, 0.75)) == pytest.approx(1 / 7)


def test_greedy_match_is_one_to_one_highest_iou_first() -> None:
    gt = [(0.0, 0.0, 0.4, 0.4), (0.05, 0.05, 0.45, 0.45)]
    pred = [(0.05, 0.05, 0.45, 0.45)]  # overlaps both GT boxes, exactly equals the 2nd
    matches = greedy_match(gt, pred, 0.5)
    assert matches == [(1, 0, pytest.approx(1.0))]
    assert greedy_match(gt, pred, 0.5) == greedy_match(gt, pred, 0.5)
    two = greedy_match(gt, [pred[0], gt[0]], 0.5)
    assert sorted((g, p) for g, p, _ in two) == [(0, 1), (1, 0)]
    assert greedy_match(gt, [(0.9, 0.9, 1.0, 1.0)], 0.5) == []


def test_frames_source_crop_and_unknown() -> None:
    """``to_source_frame``/``RegionFrameError`` stay as pure, tested geometry
    helpers even though ``region_records`` no longer wires them in -- every
    ``region_boxes`` entry is already source-frame by the time it's stored
    (see the module docstring), so there is no remaining stored per-box
    frame marker that could ever come back as ``'pixels'``."""
    box = (0.0, 0.0, 0.5, 0.5)
    assert to_source_frame(box, 'source', None) == box
    assert to_source_frame(box, None, None) == box  # pre-provenance rows
    item = (0.5, 0.5, 1.0, 1.0)
    assert to_source_frame(box, 'crop', item) == pytest.approx((0.5, 0.5, 0.75, 0.75))
    assert to_source_frame(box, 'item', item) == pytest.approx((0.5, 0.5, 0.75, 0.75))
    with pytest.raises(RegionFrameError, match='no bbox_norm'):
        to_source_frame(box, 'crop', None)
    with pytest.raises(RegionFrameError, match='unknown region bbox frame'):
        to_source_frame(box, 'pixels', item)


def test_region_records_pending_item_falls_back_to_configured_status_field() -> None:
    """A box-less item (still pending, or a terminal box-less status) has
    no ``region_boxes`` entry to read at all -- the one case where
    ``region_records`` still reads a customizable item-level field."""
    fields = RegionFields(status='roi_state')
    recs = region_records({'crop_id': 'c', 'roi_state': 'pending'}, fields)
    assert len(recs) == 1
    assert (recs[0].status, recs[0].detector, recs[0].box) == (
        RegionStatus.PENDING_DETECTION.value,
        NO_DETECTOR,
        None,
    )


@pytest.mark.parametrize(
    ('reason', 'expected'),
    [
        ('sanity_reject:degenerate_zero_size', RegionStatus.DETECTION_FAILED.value),
        ('verifier_rejected', RegionStatus.VERIFY_REJECTED.value),
        (None, RegionStatus.VERIFY_REJECTED.value),
    ],
)
def test_a_sanity_rejected_box_is_scored_as_detection_failed(
    reason: str | None, expected: str
) -> None:
    """A box a sanity gate rejected is an item-level `detection_failed`,
    not a verifier `verify_rejected` -- the two failures are different."""
    (rec,) = region_records(
        {
            'crop_id': 'c',
            'region_boxes': [
                {
                    'box_id': 'b1',
                    'bbox_norm': [0.1, 0.1, 0.2, 0.2],
                    'state': 'rejected',
                    'rejection_reason': reason,
                }
            ],
        },
        get_region_fields(),
    )
    assert rec.status == expected


def test_region_records_reads_configured_boxes_field_name() -> None:
    fields = RegionFields(boxes='roi_boxes')
    recs = region_records(
        {
            'crop_id': 'c',
            'roi_boxes': [
                {
                    'box_id': 'b1',
                    'bbox_norm': [0.1, 0.1, 0.2, 0.2],
                    'state': 'accepted',
                    'detector': 'd',
                    'score': 0.7,
                }
            ],
        },
        fields,
    )
    assert len(recs) == 1
    assert (recs[0].status, recs[0].detector, recs[0].score) == (
        RegionStatus.DETECTED.value,
        'd',
        0.7,
    )
    assert recs[0].box == pytest.approx((0.1, 0.1, 0.2, 0.2))


def test_region_records_maps_every_box_state_to_its_status() -> None:
    boxes = [
        {'box_id': 'b1', 'bbox_norm': [0, 0, 0.1, 0.1], 'state': 'accepted'},
        {'box_id': 'b2', 'bbox_norm': [0.1, 0.1, 0.2, 0.2], 'state': 'rejected'},
        {'box_id': 'b3', 'bbox_norm': [0.2, 0.2, 0.3, 0.3], 'state': 'false_positive'},
        {'box_id': 'b4', 'bbox_norm': [0.3, 0.3, 0.4, 0.4], 'state': 'proposed'},
    ]
    recs = region_records({'crop_id': 'c', F.boxes: boxes}, F)
    assert [r.status for r in recs] == [
        RegionStatus.DETECTED.value,
        RegionStatus.VERIFY_REJECTED.value,
        RegionStatus.FALSE_POSITIVE.value,
        RegionStatus.PENDING_VERIFICATION.value,
    ]


# =============================================================================
# End-to-end scoring over the fake index
# =============================================================================

GT_A1 = (0.10, 0.10, 0.20, 0.20)
GT_A2 = (0.60, 0.60, 0.70, 0.70)
GT_B = (0.40, 0.40, 0.50, 0.50)
GT_E = (0.30, 0.30, 0.40, 0.40)
GT_F = (0.20, 0.20, 0.30, 0.30)
GT_H = (0.50, 0.50, 0.75, 0.75)


# Item-level human/machine status -> the box `state` a box-list write would
# leave for that status (mirrors `RegionStatus.DETECTED` -> `accepted`,
# `VERIFY_REJECTED` -> `rejected` in the real worker; FALSE_POSITIVE shares
# its literal string with its box state already).
_STATUS_TO_BOX_STATE: dict[str, str] = {
    RegionStatus.DETECTED.value: 'accepted',
    RegionStatus.VERIFY_REJECTED.value: 'rejected',
    RegionStatus.FALSE_POSITIVE.value: RegionStatus.FALSE_POSITIVE.value,
}


def _item(crop_id: str, image_id: str, status: str | None, box: Any = None, **extra: Any):
    doc: dict[str, Any] = {
        'crop_id': crop_id,
        'image_id': image_id,
        'image_path': f'/srv/{image_id}.jpg',
        'bbox_norm': extra.pop('item_box', [0.0, 0.0, 1.0, 1.0]),
    }
    if status is not None:
        doc[F.status] = status
    detector = extra.pop('detector', 'det_a')
    score = extra.pop('score', 0.9)
    if box is not None:
        frame = extra.pop('frame', 'source')
        # `region_boxes` entries are always source-frame (every writer
        # projects before appending) -- a 'crop'/'item' frame test fixture
        # projects at construction time instead of at read time.
        stored_box = (
            list(crop_norm_to_source_norm(tuple(box), tuple(doc['bbox_norm'])))
            if frame in ('crop', 'item')
            else list(box)
        )
        doc[F.boxes] = [
            {
                'box_id': 'b1',
                'bbox_norm': stored_box,
                'state': _STATUS_TO_BOX_STATE.get(status, 'accepted') if status else 'accepted',
                'detector': detector,
                'score': score,
            }
        ]
    doc.update(extra)
    return crop_id, doc


def _fake() -> QueryFakeOpenSearch:
    images = {
        iid: {'image_id': iid, 'image_path': f'/srv/{iid}.jpg'}
        for iid in ('A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'orig')
    }
    items = dict(
        [
            _item('a1', 'A', DET, (0.11, 0.10, 0.21, 0.20)),
            _item('a2', 'A', RegionStatus.VERIFY_REJECTED.value, GT_A2),
            _item('a3', 'A', None),  # never queued for region detection
            _item('b1', 'B', DET, (0.44, 0.44, 0.54, 0.54), detector='det_b'),
            _item('c1', 'C', DET, (0.1, 0.1, 0.2, 0.2), score=0.8),
            _item('c2', 'C', DET, (0.1, 0.1, 0.2, 0.205), score=0.6),  # same region, 2nd item
            _item('d1', 'D', RegionStatus.NO_REGION_VISIBLE.value),
            _item('f1', 'F', RegionStatus.PENDING_DETECTION.value),
            _item('f2', 'F', DET, GT_F),
            _item('h1', 'H', DET, (0.0, 0.0, 0.5, 0.5), frame='crop', item_box=[0.5, 0.5, 1, 1]),
            _item('i1', 'I', 'pending'),  # legacy pending alias
            _item('o1', 'orig', DET, (0.7, 0.7, 0.8, 0.8)),
        ]
    )
    return QueryFakeOpenSearch({IMAGES: images, ITEMS: items})


def _cohort() -> list[CohortImage]:
    def c(name: str, gt: list, split: str = 'test', image_id: str | None = None) -> CohortImage:
        return CohortImage(
            key=f'/local/{name}.jpg',
            server_path=f'/srv/{name}.jpg',
            gt=list(gt),
            split=split,
            image_id=image_id,
        )

    return [
        c('A', [GT_A1, GT_A2]),
        c('B', [GT_B]),
        c('C', []),  # background, with a false positive (two items, one region)
        c('D', []),  # clean background
        c('E', [GT_E]),  # ingested, but no items on it
        c('F', [GT_F]),  # still pending
        c('G', [GT_E]),  # never ingested
        c('H', [GT_H], split='val'),  # crop-frame region
        c('I', []),
        # A content duplicate ingested under another path: joined by image_id.
        c('dup', [(0.7, 0.7, 0.8, 0.8)], split='val', image_id='orig'),
    ]


async def _run(fake: QueryFakeOpenSearch | None = None, **kw: Any):
    return await run_eval(fake or _fake(), _cohort(), config=CFG, fields=F, **kw)


@pytest.mark.asyncio
async def test_metrics_backgrounds_pending_and_frames() -> None:
    res = await _run()
    t = res.summary['total']

    assert (t['images'], t['positives'], t['backgrounds']) == (10, 7, 3)
    assert t['not_ingested'] == 1  # G
    assert (t['pending_images'], t['pending_gt_boxes']) == (2, 1)  # F + legacy-pending I
    assert (t['no_item_images'], t['no_item_gt_boxes']) == (1, 1)  # E
    # Evaluated: A(2) B(1) C(0) D(0) E(1) H(1) dup(1)
    assert (t['evaluated_images'], t['evaluated_gt_boxes']) == (7, 6)
    # a1, b1, c1 (c2 merged), h1, o1
    assert (t['predictions'], t['duplicates_merged']) == (5, 1)
    assert t['tp'] == 3  # a1, h1 (crop frame re-projected), o1 via image_id
    assert t['recall'] == pytest.approx(0.5)
    assert t['precision'] == pytest.approx(0.6)
    assert t['f1'] == pytest.approx(round(2 * 0.5 * 0.6 / 1.1, 4))
    assert t['recall_at_0.5'] == t['recall']
    assert t['false_positives'] == 2  # b1 (IoU too low), c1 (background)
    assert (t['evaluated_backgrounds'], t['backgrounds_with_detection']) == (2, 1)
    assert t['background_fp_regions'] == 1
    assert t['by_detector']['det_b'] == {'predictions': 1, 'tp': 0, 'fp': 1, 'precision': 0.0}
    assert t['by_detector']['det_a']['tp'] == 3

    assert res.summary['splits']['val']['tp'] == 2
    assert res.summary['splits']['test']['not_ingested'] == 1

    by_status = res.summary['by_status']
    assert by_status['(none)']['items'] == 1
    assert by_status[RegionStatus.PENDING_DETECTION.value]['items'] == 2
    assert by_status[DET] == {'items': 7, 'with_box': 7}
    assert res.summary['missed_but_boxed_by_status'] == {RegionStatus.VERIFY_REJECTED.value: 1}
    assert res.summary['miss_reasons'] == {
        'low_iou': 1,
        'no_items': 1,
        f'status:{RegionStatus.VERIFY_REJECTED.value}': 1,
    }
    assert res.summary['wait_timed_out'] is False

    # Worst first; the pending frame's GT box is not listed as a miss.
    assert [m['best_iou'] for m in res.misses] == sorted(m['best_iou'] for m in res.misses)
    assert res.misses[-1]['reason'] == 'low_iou'
    assert '/local/F.jpg' not in {m['image'] for m in res.misses}
    low = next(m for m in res.misses if m['reason'] == 'low_iou')
    assert low['best_iou'] == pytest.approx(iou(GT_B, (0.44, 0.44, 0.54, 0.54)), abs=1e-4)
    assert {fp['crop_id'] for fp in res.false_positives} == {'b1', 'c1'}
    assert next(fp for fp in res.false_positives if fp['crop_id'] == 'c1')['background']


@pytest.mark.asyncio
async def test_iou_threshold_and_accepted_statuses_are_configurable() -> None:
    loose = await _run(iou_threshold=0.1)
    assert loose.summary['total']['tp'] == 4  # b1 now matches
    assert loose.summary['total']['recall_at_0.5'] == pytest.approx(0.5)

    wide = await _run(accepted=[DET, RegionStatus.VERIFY_REJECTED.value])
    assert wide.summary['total']['tp'] == 4  # a2's rejected box now counts
    assert wide.summary['missed_but_boxed_by_status'] == {}


@pytest.mark.asyncio
async def test_dedup_can_be_disabled() -> None:
    res = await _run(dedup_iou=0.0)
    t = res.summary['total']
    assert (t['duplicates_merged'], t['background_fp_regions']) == (0, 2)


@pytest.mark.asyncio
async def test_wait_pending_polls_until_drained() -> None:
    fake = _fake()
    sleeps: list[float] = []

    async def _sleep(s: float) -> None:
        sleeps.append(s)
        fake.store[ITEMS]['f1'][F.status] = RegionStatus.NO_REGION_BOX.value
        fake.store[ITEMS]['i1'][F.status] = RegionStatus.NO_REGION_VISIBLE.value

    res = await _run(fake, wait_pending_s=100, poll_interval_s=5, sleep=_sleep)
    assert sleeps == [5]
    t = res.summary['total']
    assert t['pending_images'] == 0
    assert t['tp'] == 4  # F's detected region now scored
    assert res.summary['wait_timed_out'] is False


@pytest.mark.asyncio
async def test_wait_pending_times_out_with_pending_reported() -> None:
    now = [0.0]

    async def _sleep(s: float) -> None:
        now[0] += s

    res = await _run(wait_pending_s=12, poll_interval_s=5, sleep=_sleep, clock=lambda: now[0])
    assert now[0] == pytest.approx(12)
    assert res.summary['total']['pending_images'] == 2
    assert res.summary['wait_timed_out'] is True


def test_evaluate_all_backgrounds_has_no_recall() -> None:
    res = evaluate([CohortImage(key='k', server_path='k', gt=[], image_id='x')], {})
    t = res.summary['total']
    assert (t['recall'], t['precision'], t['f1']) == (None, None, None)
    assert t['no_item_images'] == 1


# =============================================================================
# CLI
# =============================================================================


def _cli():
    from scripts.curation import eval_regions_vs_gt

    return eval_regions_vs_gt


class _Closable:
    def __init__(self, inner: QueryFakeOpenSearch) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def close(self) -> None:
        return None


def _dataset(root: Path) -> Path:
    def _w(name: str, label: str | None) -> None:
        img = root / 'images' / 'test' / f'{name}.jpg'
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b'x')
        if label is not None:
            lbl = root / 'labels' / 'test' / f'{name}.txt'
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text(label)

    _w('hit', '0 0.15 0.15 0.1 0.1\n')
    _w('bg', '')
    _w('nolabel', None)
    data = root / 'data.yaml'
    data.write_text('test: images/test\nnc: 1\nnames: {0: region}\n')
    return data


def _cli_fake(root: Path) -> QueryFakeOpenSearch:
    srv = '/server/ds/images/test'
    return QueryFakeOpenSearch(
        {
            IMAGES: {
                'h': {'image_id': 'h', 'image_path': f'{srv}/hit.jpg'},
                'b': {'image_id': 'b', 'image_path': f'{srv}/bg.jpg'},
            },
            ITEMS: dict(
                [
                    _item('x1', 'h', DET, (0.1, 0.1, 0.2, 0.2)),
                    _item('x2', 'b', DET, (0.5, 0.5, 0.6, 0.6)),
                ]
            ),
        }
    )


def _run_cli(monkeypatch, fake: QueryFakeOpenSearch, argv: list[str]) -> int:
    mod = _cli()
    monkeypatch.setattr(mod, 'make_script_opensearch', lambda *_a, **_kw: _Closable(fake))
    monkeypatch.setattr(mod, 'get_curation_config', lambda: CFG)
    return mod.main(argv)


def _dataset_with_duplicated_split_dir(root: Path) -> Path:
    """A data.yaml where train: and val: both point at images/validation --
    the F-76 bug shape (originally produced by
    scripts/datasets/fetch_openimages_plates.py)."""
    img = root / 'images' / 'validation' / 'a.jpg'
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b'x')
    lbl = root / 'labels' / 'validation' / 'a.txt'
    lbl.parent.mkdir(parents=True, exist_ok=True)
    lbl.write_text('0 0.15 0.15 0.1 0.1\n')
    data = root / 'data.yaml'
    data.write_text('train: images/validation\nval: images/validation\nnc: 1\nnames: {0: region}\n')
    return data


def test_build_cohort_deduplicates_when_a_yaml_lists_the_same_dir_twice(
    tmp_path: Path,
) -> None:
    """F-76: train: and val: both resolving to images/validation must not
    make build_cohort() count and score the same image twice."""
    mod = _cli()
    data = _dataset_with_duplicated_split_dir(tmp_path / 'ds')
    args = argparse.Namespace(
        state_dir=None,
        image_list=None,
        dataset=data,
        splits=None,
        path_map=None,
        gt_class=None,
    )
    cohort = mod.build_cohort(args)
    assert len(cohort) == 1
    assert {c.split for c in cohort} == {'train'}  # first occurrence wins


def test_cli_dataset_cohort_with_path_map(monkeypatch, tmp_path: Path, capsys) -> None:
    data = _dataset(tmp_path / 'ds')
    out = tmp_path / 'out'
    argv = [
        '--dataset',
        str(data),
        '--path-map',
        f'{tmp_path / "ds"}=/server/ds',
        '--out-dir',
        str(out),
    ]
    assert _run_cli(monkeypatch, _cli_fake(tmp_path), argv) == 0
    summary = json.loads((out / 'summary.json').read_text())
    t = summary['total']
    assert (t['images'], t['not_ingested'], t['tp'], t['background_fp_regions']) == (3, 1, 1, 1)
    assert t['recall'] == 1.0
    fps = [json.loads(ln) for ln in (out / 'false_positives.jsonl').read_text().splitlines()]
    assert [fp['crop_id'] for fp in fps] == ['x2']
    assert (out / 'misses.jsonl').read_text() == ''
    printed = capsys.readouterr().out
    assert 'R@0.5' in printed
    assert 'det_a' in printed


def test_cli_state_dir_cohort_uses_recorded_ids(monkeypatch, tmp_path: Path) -> None:
    data = _dataset(tmp_path / 'ds')
    state = tmp_path / 'state'
    (state / 'ingested').mkdir(parents=True)
    img = tmp_path / 'ds' / 'images' / 'test' / 'hit.jpg'
    (state / 'ingested' / 'test.jsonl').write_text(
        json.dumps({'image': str(img), 'server_path': '/elsewhere/hit.jpg', 'image_id': 'h'}) + '\n'
    )
    out = tmp_path / 'out'
    argv = ['--dataset', str(data), '--state-dir', str(state), '--out-dir', str(out)]
    assert _run_cli(monkeypatch, _cli_fake(tmp_path), argv) == 0
    t = json.loads((out / 'summary.json').read_text())['total']
    assert (t['images'], t['tp'], t['predictions']) == (1, 1, 1)

    assert _run_cli(monkeypatch, _cli_fake(tmp_path), [*argv, '--splits', 'train']) == 1


# test_cli_unknown_frame_exits_3 (and test_unknown_frame_is_refused above)
# were removed in the W8-cleanup port: both simulated a stored
# `region_bbox_frame='pixels'` on an item to exercise the CLI's
# RegionFrameError -> exit(3) path. `region_boxes` entries have no stored
# per-box frame marker at all (every writer projects to source frame
# before appending -- see the region_eval module docstring), so that
# scenario is now structurally impossible to construct; RegionFrameError
# and the CLI's exit(3) branch remain as defensive code (still directly
# unit-tested via to_source_frame in test_frames_source_crop_and_unknown)
# but have no live end-to-end trigger left to test.


# =============================================================================
# --import-id cohort (W10)
# =============================================================================

SERVER = '/server/ds/images/test'


def _entries() -> list[dict[str, Any]]:
    ok = {'status': 'ok', 'split': 'test'}
    return [
        {
            **ok,
            'rel_path': 'images/test/hit.jpg',
            'image_id': 'h',
            'image_path': f'{SERVER}/hit.jpg',
        },
        {**ok, 'rel_path': 'images/test/bg.jpg', 'image_id': 'b', 'image_path': f'{SERVER}/bg.jpg'},
        # an entry the import failed on, and one that never got an image: neither is scored
        {
            'status': 'failed',
            'split': 'test',
            'rel_path': 'images/test/nolabel.jpg',
            'image_id': None,
        },
        {**ok, 'rel_path': 'images/test/gone.jpg', 'image_id': None},
    ]


def test_import_cohort_scores_exactly_the_imported_frames(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    mod = _cli()
    data = _dataset(tmp_path / 'ds')
    seen: dict[str, Any] = {}

    def fake_fetch(api, project, import_id, splits):
        seen.update(api=api, project=project, import_id=import_id, splits=splits)
        return _entries()

    monkeypatch.setattr(mod, 'fetch_import_entries', fake_fetch)
    out = tmp_path / 'out'
    argv = [
        '--dataset', str(data), '--import-id', 'imp_1', '--splits', 'test',
        '--api', 'http://api.test', '--out-dir', str(out),
    ]  # fmt: skip
    assert _run_cli(monkeypatch, _cli_fake(tmp_path), argv) == 0
    assert seen == {
        'api': 'http://api.test',
        'project': 'default',
        'import_id': 'imp_1',
        'splits': ['test'],
    }
    t = json.loads((out / 'summary.json').read_text())['total']
    # 2 written frames (the failed and the image-less entries are not frames)
    assert (t['images'], t['not_ingested'], t['tp'], t['background_fp_regions']) == (2, 0, 1, 1)
    assert t['recall'] == 1.0


@pytest.mark.parametrize('rel_path', ['../outside/frame.jpg', '/etc/passwd'])
def test_an_import_entry_outside_the_dataset_root_is_refused(tmp_path: Path, rel_path: str) -> None:
    mod = _cli()
    data = _dataset(tmp_path / 'ds')
    entry = {'status': 'written', 'split': 'test', 'image_id': 'x', 'rel_path': rel_path}
    with pytest.raises(mod.DatasetError, match='outside the dataset'):
        mod.cohort_from_import(
            [entry], data.parent if data.is_file() else data, path_map=None, class_ids=None
        )


def test_import_cohort_with_nothing_written_is_a_clear_error(monkeypatch, tmp_path: Path) -> None:
    mod = _cli()
    data = _dataset(tmp_path / 'ds')
    monkeypatch.setattr(mod, 'fetch_import_entries', lambda *_a: [{'status': 'failed'}])
    argv = ['--dataset', str(data), '--import-id', 'imp_1', '--out-dir', str(tmp_path / 'o')]
    assert _run_cli(monkeypatch, _cli_fake(tmp_path), argv) == 1


def test_fetch_import_entries_pages_and_filters_by_split(monkeypatch) -> None:
    mod = _cli()
    calls: list[dict[str, Any]] = []

    class _Resp:
        status_code = 200

        def __init__(self, body: dict[str, Any]) -> None:
            self._body = body

        def json(self) -> dict[str, Any]:
            return self._body

    def fake_get(url, params, timeout):
        calls.append({'url': url, **params})
        return _Resp(
            {
                'items': [{'rel_path': f'{params["split"]}{params["page"]}'}],
                'total': 501 if params['split'] == 'train' else 1,
            }
        )

    import requests

    monkeypatch.setattr(requests, 'get', fake_get)
    rows = mod.fetch_import_entries('http://api.test/', 'p', 'imp_1', ['train', 'test'])
    assert [r['rel_path'] for r in rows] == ['train1', 'train2', 'test1']
    assert calls[0]['url'] == 'http://api.test/curation/projects/p/datasets/imports/imp_1/entries'
    assert [c['page'] for c in calls] == [1, 2, 1]
