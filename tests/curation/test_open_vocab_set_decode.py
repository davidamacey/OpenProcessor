"""The one decoder of a stored open-vocabulary body."""

from __future__ import annotations

import pytest

from src.services.detection.open_vocab_set import decode_open_vocab_set


def test_defaults_and_nested_gating() -> None:
    ov = decode_open_vocab_set(
        's',
        {
            'targets': [{'prompt': 'cup', 'class_name': 'cup', 'parent_classes': ['table']}],
            'gating': {'tier3_hit_rate': {'enabled': True, 'window': 10}},
        },
    )
    target = ov.targets[0]
    assert (target.min_score, target.max_area_frac, target.max_instances) == (0.5, 0.9, 20)
    assert target.parent_classes == ('table',)
    assert ov.run_on_ingest is False
    assert ov.max_enabled_targets == 8
    assert ov.gating.tier2_vlm_precheck is False
    assert (ov.gating.tier3_hit_rate.enabled, ov.gating.tier3_hit_rate.window) == (True, 10)
    assert ov.gating.tier3_hit_rate.miss_threshold == 15
    assert target.rules().class_name == 'cup'


@pytest.mark.parametrize(
    ('body', 'fragment'),
    [
        ({'bogus': 1}, "unknown field 'bogus'"),
        ({'targets': [{'prompt': 'a', 'min_scor': 0.5}]}, 'targets[0]: unknown field'),
        ({'targets': [{'prompt': 5}]}, 'targets[0].prompt: expected str'),
        ({'targets': [{'prompt': 'a', 'enabled': 1}]}, 'enabled: expected bool'),
        ({'targets': 'x'}, 'body.targets: expected a list'),
        ({'targets': [{'prompt': 'a', 'parent_classes': 'car'}]}, 'list of strings'),
        ({'image_max_side': 10.5}, 'expected int'),
        ({'gating': {'tier3_hit_rate': {'nope': 1}}}, 'unknown field'),
    ],
)
def test_malformed_bodies_fail_loudly(body: dict, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment.replace('[', r'\[').replace(']', r'\]')):
        decode_open_vocab_set('s', body)


def test_an_int_is_accepted_where_a_float_is_expected() -> None:
    ov = decode_open_vocab_set('s', {'dedup_iou': 1, 'targets': [{'prompt': 'a', 'min_score': 1}]})
    assert ov.dedup_iou == 1.0
    assert isinstance(ov.targets[0].min_score, float)
