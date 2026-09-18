"""Pins for ``DetectionProfile`` (Chunk 0).

See ``docs/design/oss_genericization_phase2_plan.md`` §3.3.
"""

from __future__ import annotations

import re

from src.config import DetectionProfile


def test_requires_a_name_only() -> None:
    profile = DetectionProfile(name='license_plate')
    assert profile.name == 'license_plate'


def test_defaults_match_reference_constants() -> None:
    profile = DetectionProfile(name='license_plate')
    assert profile.input_size == 640
    assert profile.confidence_floor == 0.4
    assert profile.letterbox_fill == (114, 114, 114)
    assert profile.batch_limit == 16
    assert profile.segmenter_name == 'sam3'
    assert profile.segmenter_version == '1'
    assert profile.ocr_det_model == 'paddleocr_det_trt'
    assert profile.ocr_pipeline_model == 'ocr_pipeline'
    assert profile.aspect_min == 1.2
    assert profile.aspect_max == 8.0


def test_text_pattern_is_a_plain_str_not_compiled() -> None:
    """A frozen dataclass holding a compiled re.Pattern is not cleanly
    serializable and breaks equality in tests — text_pattern must stay a
    str; consumers compile it once at module scope.
    """
    profile = DetectionProfile(name='license_plate')
    assert isinstance(profile.text_pattern, str)
    # Sanity: it is still a usable regex once compiled by the consumer.
    compiled = re.compile(profile.text_pattern)
    assert compiled.match('AB1234')


def test_is_frozen() -> None:
    profile = DetectionProfile(name='license_plate')
    try:
        profile.name = 'mutated'  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError('DetectionProfile must be immutable (frozen dataclass)')


def test_two_distinct_profiles_are_independent() -> None:
    """A second, non-plate profile through the same cascade must not
    share mutable state with the first (§6.1 wave-8 coverage note)."""
    plate = DetectionProfile(name='license_plate', detector_model='lpr_nanov11_640')
    box = DetectionProfile(
        name='box',
        detector_model='box_detector_v1',
        aspect_min=0.5,
        aspect_max=2.0,
        secondary_shape_groups=frozenset({'small_box', 'large_box'}),
    )
    assert plate.detector_model != box.detector_model
    assert plate.aspect_min != box.aspect_min
    assert box.secondary_shape_groups == frozenset({'small_box', 'large_box'})
    assert plate.secondary_shape_groups == frozenset()
