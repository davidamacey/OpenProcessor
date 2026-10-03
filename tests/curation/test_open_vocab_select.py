"""Wave 2: the pure select + dedup function of the full-image SAM 3 pass."""

from __future__ import annotations

import pytest

from src.services.detection.cascade_detect import RegionCandidate
from src.services.detection.open_vocab_select import (
    ExistingBox,
    TargetRules,
    select_open_vocab_hits,
)


def _c(box: tuple[float, float, float, float], score: float = 0.9) -> RegionCandidate:
    return RegionCandidate(bbox_norm=box, score=score, source='sam3')


def _rules(prompt: str = 'cup', cls: str = 'cup', **kw: float) -> TargetRules:
    base = {
        'min_score': 0.5,
        'min_area_frac': 0.001,
        'max_area_frac': 0.9,
        'max_instances': 20,
    }
    base.update(kw)
    return TargetRules(prompt=prompt, class_name=cls, **base)  # type: ignore[arg-type]


BOX = (0.1, 0.1, 0.3, 0.3)


def _reasons(sel) -> list[str]:
    return [r for _, r in sel.dropped]


def test_a_plain_hit_is_kept_with_its_target() -> None:
    sel = select_open_vocab_hits([(_rules(), [_c(BOX)])], [], dedup_iou=0.5)
    assert [(h.prompt, h.class_name) for h in sel.kept] == [('cup', 'cup')]
    assert sel.dropped == []


@pytest.mark.parametrize(
    ('cand', 'rules_kw', 'reason'),
    [
        (_c(BOX, 0.4), {}, 'below_min_score'),
        (_c(BOX, 0.5), {'min_score': 0.5}, None),
        (_c((0.1, 0.1, 0.11, 0.11)), {'min_area_frac': 0.01}, 'too_small'),
        (_c((0.0, 0.0, 1.0, 1.0)), {'max_area_frac': 0.9}, 'too_large'),
        (_c((0.0, 0.0, 0.9, 1.0)), {'max_area_frac': 0.9}, None),
    ],
)
def test_per_target_filters(cand, rules_kw, reason) -> None:
    sel = select_open_vocab_hits([(_rules(**rules_kw), [cand])], [], dedup_iou=0.5)
    assert _reasons(sel) == ([reason] if reason else [])
    assert len(sel.kept) == (0 if reason else 1)


def test_nms_and_instance_cap() -> None:
    cands = [
        _c((0.1, 0.1, 0.3, 0.3), 0.9),
        _c((0.11, 0.1, 0.31, 0.3), 0.8),
        _c((0.5, 0.5, 0.6, 0.6), 0.7),
        _c((0.7, 0.7, 0.8, 0.8), 0.6),
    ]
    sel = select_open_vocab_hits([(_rules(max_instances=2), cands)], [], dedup_iou=0.5)
    assert [h.candidate.score for h in sel.kept] == [0.9, 0.7]
    assert sorted(_reasons(sel)) == ['nms', 'over_max']


def test_same_class_from_two_targets_is_merged_but_different_class_is_not() -> None:
    a = (_rules('mug', 'cup'), [_c(BOX, 0.9)])
    b = (_rules('cup', 'cup'), [_c(BOX, 0.8)])
    sel = select_open_vocab_hits([a, b], [], dedup_iou=0.5)
    assert [h.prompt for h in sel.kept] == ['mug']
    assert _reasons(sel) == ['cross_target_nms']

    c = (_rules('bottle', 'bottle'), [_c(BOX, 0.8)])
    sel = select_open_vocab_hits([a, c], [], dedup_iou=0.5)
    assert len(sel.kept) == 2


def test_same_class_existing_box_wins() -> None:
    existing = [ExistingBox(BOX, 'Cup ', locked=False)]
    sel = select_open_vocab_hits([(_rules(), [_c(BOX)])], existing, dedup_iou=0.5)
    assert sel.kept == []
    assert _reasons(sel) == ['agree_existing']


def test_different_class_machine_box_keeps_both() -> None:
    existing = [ExistingBox(BOX, 'bowl', locked=False)]
    sel = select_open_vocab_hits([(_rules(), [_c(BOX)])], existing, dedup_iou=0.5)
    assert len(sel.kept) == 1


def test_locked_box_blocks_any_class_at_high_iou_only() -> None:
    near = (0.1, 0.1, 0.3, 0.29)  # IoU ~0.95 with BOX
    far = (0.1, 0.1, 0.3, 0.2)  # IoU 0.5 with BOX
    existing = [ExistingBox(BOX, 'bowl', locked=True)]
    assert _reasons(select_open_vocab_hits([(_rules(), [_c(near)])], existing, dedup_iou=0.5)) == [
        'skipped_locked'
    ]
    sel = select_open_vocab_hits([(_rules(), [_c(far)])], existing, dedup_iou=0.9)
    assert len(sel.kept) == 1


def test_iou_boundary_is_inclusive_for_dedup() -> None:
    # IoU of these two boxes is exactly 0.5.
    existing = [ExistingBox((0.0, 0.0, 0.2, 0.2), 'cup', locked=False)]
    hit = _c((0.0, 0.0, 0.2, 0.1))
    sel = select_open_vocab_hits([(_rules(min_area_frac=0.0), [hit])], existing, dedup_iou=0.5)
    assert _reasons(sel) == ['agree_existing']


def test_deterministic_regardless_of_input_order() -> None:
    cands = [_c((0.1, 0.1, 0.2, 0.2), 0.7), _c((0.5, 0.5, 0.6, 0.6), 0.7)]
    one = select_open_vocab_hits([(_rules(), cands)], [], dedup_iou=0.5)
    two = select_open_vocab_hits([(_rules(), cands[::-1])], [], dedup_iou=0.5)
    assert [h.candidate for h in one.kept] == [h.candidate for h in two.kept]


def test_class_names_compare_by_the_one_name_equality_rule() -> None:
    existing = [ExistingBox(BOX, 'traffic_light', locked=False)]
    rules = _rules('lamp', 'Traffic Light')
    sel = select_open_vocab_hits([(rules, [_c(BOX)])], existing, dedup_iou=0.5)
    assert _reasons(sel) == ['agree_existing']
