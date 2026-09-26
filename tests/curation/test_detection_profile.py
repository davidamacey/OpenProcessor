"""Pins for ``DetectionProfile``.

See ``docs/design/curation_design_rationale.md`` §2.3.
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


def test_ocr_rec_model_defaults_to_the_real_triton_directory_name() -> None:
    """This used to default to 'paddleocr_rec', but no model
    directory of that name exists -- the real one (verified against
    docker-compose.yml's --load-model list, scripts/setup.sh's
    required_models, scripts/export_paddleocr.sh, and the models/ tree
    on disk) is 'paddleocr_rec_trt'. GET /curation/models/status queries
    Triton by this name; the old default made it permanently report a
    missing model."""
    profile = DetectionProfile(name='license_plate')
    assert profile.ocr_rec_model == 'paddleocr_rec_trt'


def test_from_env_overrides_every_field(monkeypatch) -> None:
    from dataclasses import fields

    env_values = {
        'NAME': 'env_region',
        'DETECTOR_MODEL': 'env_detector',
        'DETECTOR_VERSION': '2',
        'INPUT_SIZE': '512',
        'CONFIDENCE_FLOOR': '0.55',
        'BATCH_LIMIT': '32',
        'LETTERBOX_FILL': '10,20,30',
        'FEATURE_OUTPUT': 'env_feature',
        'ASPECT_MIN': '0.9',
        'ASPECT_MAX': '3.3',
        'TEXT_HINT_ASPECT_MIN': '1.1',
        'TEXT_HINT_ASPECT_MAX': '5.5',
        'TEXT_HINT_REC_FLOOR': '0.65',
        'TEXT_HINT_LEN_MIN': '3',
        'TEXT_HINT_LEN_MAX': '12',
        'TEXT_HINT_ENABLED': 'false',
        'TEXT_HINT_REQUIRE_LETTERS_AND_DIGITS': 'true',
        'AUTO_CONFIRM_ASPECT': '0.25,4.5',
        'AUTO_CONFIRM_AREA_FRAC': '0.01,0.75',
        'TEXT_PATTERN': r'[0-9]{3,}',
        'SEGMENTER_NAME': 'env_segmenter',
        'SEGMENTER_VERSION': '3',
        'HUMAN_DETECTOR_NAME': 'env_human',
        'HUMAN_DETECTOR_VERSION': '4',
        'OCR_DET_MODEL': 'env_ocr_det',
        'OCR_DET_VERSION': '5',
        'OCR_DET_INPUT_SIZE': '480',
        'OCR_DET_PROB_FLOOR': '0.42',
        'OCR_REC_MODEL': 'env_ocr_rec',
        'OCR_REC_VERSION': '6',
        'OCR_PIPELINE_MODEL': 'env_ocr_pipeline',
        'TEXT_READER': 'both',
        'TEXT_CROP_MARGIN': '0.02',
        'TEXT_CROP_MIN_HEIGHT': '72',
        'TEXT_MIN_HEIGHT_RATIO': '0.55',
        'TEXT_BORDER_MARGIN': '0.1',
        'TEXT_UPPERCASE': 'true',
        'TEXT_CHARSET': '[A-Z0-9]',
        'TEXT_JOIN': '-',
        'TEXT_LEN_MIN': '3',
        'TEXT_LEN_MAX': '9',
        'TEXT_STOPWORDS': 'alpha, beta',
        'TEXT_MIN_CONFIDENCE': '0.4',
        'TEXT_FORMAT': '[A-Z]{3}[0-9]{3}',
        'TEXT_PLACEHOLDERS': 'XX11, YY22',
        'TEXT_REJECT_SEQUENCES': 'true',
        'SEGMENTER_TEXT_PROMPT': 'env prompt',
        'SECONDARY_SHAPE_GROUPS': 'group_a,group_b',
        'CLASS_IDS': '2, 3,7',
        'PARENT_CLASSES': 'car, Bus',
        'ASSIGNS_CLASS': 'true',
        'LABELS_PATH': '/models/proposer/labels.txt',
        'REGION_CLASS_NAME': 'env_region_class',
        'DISPLAY_NAME': 'Env Regions',
        'DISPLAY_NAME_SINGULAR': 'Env Region',
    }
    prefix = 'OP_TEST_DETECTION_'
    for suffix, value in env_values.items():
        monkeypatch.setenv(f'{prefix}{suffix}', value)

    profile = DetectionProfile.from_env(prefix)

    # Every field the dataclass declares got an env override -- fail
    # loudly (rather than silently) if a future field is added here
    # without a matching env_values entry above.
    field_names = {f.name for f in fields(profile)}
    assert field_names == {name.lower() for name in env_values}

    assert profile.name == 'env_region'
    assert profile.detector_model == 'env_detector'
    assert profile.detector_version == '2'
    assert profile.input_size == 512
    assert profile.confidence_floor == 0.55
    assert profile.batch_limit == 32
    assert profile.letterbox_fill == (10, 20, 30)
    assert profile.feature_output == 'env_feature'
    assert profile.aspect_min == 0.9
    assert profile.aspect_max == 3.3
    assert profile.text_hint_aspect_min == 1.1
    assert profile.text_hint_aspect_max == 5.5
    assert profile.text_hint_rec_floor == 0.65
    assert profile.text_hint_len_min == 3
    assert profile.text_hint_len_max == 12
    assert profile.text_hint_enabled is False
    assert profile.text_hint_require_letters_and_digits is True
    assert profile.parent_classes == frozenset({'car', 'Bus'})
    assert profile.auto_confirm_aspect == (0.25, 4.5)
    assert profile.auto_confirm_area_frac == (0.01, 0.75)
    assert profile.text_pattern == r'[0-9]{3,}'
    assert profile.segmenter_name == 'env_segmenter'
    assert profile.segmenter_version == '3'
    assert profile.human_detector_name == 'env_human'
    assert profile.human_detector_version == '4'
    assert profile.ocr_det_model == 'env_ocr_det'
    assert profile.ocr_det_version == '5'
    assert profile.ocr_det_input_size == 480
    assert profile.ocr_det_prob_floor == 0.42
    assert profile.ocr_rec_model == 'env_ocr_rec'
    assert profile.ocr_rec_version == '6'
    assert profile.ocr_pipeline_model == 'env_ocr_pipeline'
    assert profile.text_reader == 'both'
    assert profile.text_crop_margin == 0.02
    assert profile.text_crop_min_height == 72
    assert profile.text_min_height_ratio == 0.55
    assert profile.text_border_margin == 0.1
    assert profile.text_uppercase is True
    assert profile.text_charset == '[A-Z0-9]'
    assert profile.text_join == '-'
    assert profile.text_len_min == 3
    assert profile.text_len_max == 9
    assert profile.text_stopwords == frozenset({'alpha', 'beta'})
    assert profile.text_min_confidence == 0.4
    assert profile.text_format == '[A-Z]{3}[0-9]{3}'
    assert profile.text_placeholders == frozenset({'XX11', 'YY22'})
    assert profile.text_reject_sequences is True
    assert profile.segmenter_text_prompt == 'env prompt'
    assert profile.secondary_shape_groups == frozenset({'group_a', 'group_b'})
    assert profile.class_ids == frozenset({2, 3, 7})
    assert profile.assigns_class is True
    assert profile.labels_path == '/models/proposer/labels.txt'
    assert profile.region_class_name == 'env_region_class'
    assert profile.display_name == 'Env Regions'
    assert profile.display_name_singular == 'Env Region'


def test_region_class_name_and_display_name_default_empty() -> None:
    profile = DetectionProfile(name='region')
    assert profile.region_class_name == ''
    assert profile.display_name == ''
    assert profile.display_name_singular == ''


def test_from_env_overrides_only_set_vars_others_default(monkeypatch) -> None:
    monkeypatch.setenv('OP_TEST_DETECTION_ASPECT_MIN', '0.1')
    profile = DetectionProfile.from_env('OP_TEST_DETECTION_')
    assert profile.aspect_min == 0.1
    # Unset vars fall back to the dataclass default.
    assert profile.name == 'region'
    assert profile.ocr_rec_model == 'paddleocr_rec_trt'
    assert profile.segmenter_name == 'sam3'


def test_from_env_name_kwarg_used_when_name_env_unset() -> None:
    profile = DetectionProfile.from_env('OP_TEST_DETECTION_', name='custom_default_name')
    assert profile.name == 'custom_default_name'


def test_two_distinct_profiles_are_independent() -> None:
    """A second, distinct profile through the same cascade must not
    share mutable state with the first."""
    region_a = DetectionProfile(name='license_plate', detector_model='region_det_test')
    box = DetectionProfile(
        name='box',
        detector_model='box_detector_v1',
        aspect_min=0.5,
        aspect_max=2.0,
        secondary_shape_groups=frozenset({'small_box', 'large_box'}),
    )
    assert region_a.detector_model != box.detector_model
    assert region_a.aspect_min != box.aspect_min
    assert box.secondary_shape_groups == frozenset({'small_box', 'large_box'})
    assert region_a.secondary_shape_groups == frozenset()
