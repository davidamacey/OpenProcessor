"""Shared, domain-neutral :class:`DetectionProfile` fixtures for tests.

``NEUTRAL_REGION_PROFILE`` mirrors a fully-populated profile's numeric
shape (the same aspect bands / OCR thresholds a real deployment would
tune) with no domain-specific naming, for tests that need *a* realistic
profile but are not testing the ``license_plate`` example specifically.

``EXAMPLE_LICENSE_PLATE_PROFILE`` loads the actual worked example the
app ships in ``examples/region_profiles/license_plate.json`` -- only
tests that exercise that example file itself should use it.
"""

from __future__ import annotations

from pathlib import Path

from src.config import DetectionProfile
from src.services.detection.profile_registry import region_profile_from_file


_REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_LICENSE_PLATE_PROFILE_PATH = str(
    _REPO_ROOT / 'examples' / 'region_profiles' / 'license_plate.json'
)
EXAMPLE_LICENSE_PLATE_PROFILE: DetectionProfile = region_profile_from_file(
    EXAMPLE_LICENSE_PLATE_PROFILE_PATH
)
# The example profile runs segmenter-only (empty detector_model); tests that
# drive the detector leg on it configure this model on top via
# OP_REGION_DETECTION_DETECTOR_MODEL, as a deployment with its own detector
# would (the ``reference_region_profile`` fixture does).
REFERENCE_REGION_DETECTOR_MODEL = 'license_plate_detector'

NEUTRAL_REGION_PROFILE = DetectionProfile(
    name='region',
    detector_model='region_detector_test',
    detector_version='1',
    input_size=640,
    confidence_floor=0.4,
    batch_limit=16,
    letterbox_fill=(114, 114, 114),
    aspect_min=1.2,
    aspect_max=8.0,
    text_hint_aspect_min=1.5,
    text_hint_aspect_max=7.0,
    text_hint_rec_floor=0.70,
    text_hint_len_min=4,
    text_hint_len_max=10,
    segmenter_name='segmenter_test',
    segmenter_version='1',
    human_detector_name='human',
    human_detector_version='1',
    ocr_det_model='paddleocr_det_trt',
    ocr_det_version='1',
    ocr_det_input_size=640,
    ocr_det_prob_floor=0.30,
    ocr_rec_model='paddleocr_rec_trt',
    ocr_rec_version='1',
    ocr_pipeline_model='ocr_pipeline',
    text_reader='both',
    text_crop_margin=0.0,
    text_crop_min_height=56,
    text_min_height_ratio=0.6,
    text_border_margin=0.08,
    text_uppercase=True,
    text_charset='[A-Z0-9]',
    text_join='',
    text_len_min=2,
    text_len_max=10,
    # Not domain-secret: issuer names printed on a real license plate are
    # public geographic data, kept here so tests exercising the
    # stopword-removal mechanism (not the example profile) still have
    # realistic content to remove.
    text_stopwords=frozenset({'NEW YORK', 'TEXAS', 'CALIFORNIA'}),
    text_min_confidence=0.6,
    text_reject_sequences=True,
    segmenter_text_prompt='test region, test marker',
    secondary_shape_groups=frozenset(),
    region_class_name='region',
    display_name='Regions',
)
