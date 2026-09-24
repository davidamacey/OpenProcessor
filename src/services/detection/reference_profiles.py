"""Built-in *reference* :class:`~src.config.DetectionProfile` instances.

These are worked examples of a fully-populated region profile, selectable
by name (``OP_REGION_PROFILE=<name>``, see
:mod:`src.services.detection.profile_registry`). None of them is active
by default: an unconfigured deployment runs with **no** region profile,
and region detection stays off until one is selected or configured.

``REFERENCE_LICENSE_PLATE_PROFILE`` reproduces the constants of the
original reference license-plate deployment. It is an example of the
shape a profile takes, not a suggested starting point for a different
region type.
"""

from __future__ import annotations

from types import MappingProxyType

from src.config import DetectionProfile


REFERENCE_LICENSE_PLATE_PROFILE = DetectionProfile(
    name='license_plate',
    # D9: no hardcoded proprietary Triton model id. A neutral example id;
    # a real deployment sets OP_REGION_DETECTION_DETECTOR_MODEL to its
    # actual Triton model name.
    detector_model='license_plate_detector',
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
    segmenter_name='sam3',
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
    # Region-text reader. 'both' stores the VLM and OCR readings side by
    # side and flags disagreement: the OCR read costs ~50 ms per region
    # (one OCR-pipeline call) against a multi-second VLM call, and the
    # cross-check is the cheapest source of text-review candidates.
    text_reader='both',
    # Calibrated on 207 reference region crops (median 39x25 px): no extra
    # margin reads best -- detector boxes already include the frame, and
    # a margin pulls in neighboring text. Crops are framed at 56 px tall
    # for the OCR detector (see PaddleOcrTextRecognizer.read_region_lines).
    text_crop_margin=0.0,
    text_crop_min_height=56,
    text_min_height_ratio=0.6,
    text_border_margin=0.08,
    text_uppercase=True,
    text_charset='[A-Z0-9]',
    text_join='',
    text_len_min=2,
    text_len_max=10,
    # On the calibration set every reading whose weakest line scored
    # below 0.6 disagreed with the reference reading (37 of 151 readings,
    # none an exact or near match); at >= 0.6, 78 of 114 were exact or
    # within one edit-distance-ish of it (similarity >= 0.8).
    text_min_confidence=0.6,
    # A run like "123" / "999" / "XYZ" is a stock non-answer, not a reading.
    text_reject_sequences=True,
    # Issuer names printed on the region; dropped as whole lines (or as
    # single words inside a line) before the dominant line is chosen.
    text_stopwords=frozenset(
        {
            'USA',
            'ALABAMA',
            'ALASKA',
            'ARIZONA',
            'ARKANSAS',
            'CALIFORNIA',
            'COLORADO',
            'CONNECTICUT',
            'DELAWARE',
            'FLORIDA',
            'GEORGIA',
            'HAWAII',
            'IDAHO',
            'ILLINOIS',
            'INDIANA',
            'IOWA',
            'KANSAS',
            'KENTUCKY',
            'LOUISIANA',
            'MAINE',
            'MARYLAND',
            'MASSACHUSETTS',
            'MICHIGAN',
            'MINNESOTA',
            'MISSISSIPPI',
            'MISSOURI',
            'MONTANA',
            'NEBRASKA',
            'NEVADA',
            'NEW HAMPSHIRE',
            'NEW JERSEY',
            'NEW MEXICO',
            'NEW YORK',
            'NORTH CAROLINA',
            'NORTH DAKOTA',
            'OHIO',
            'OKLAHOMA',
            'OREGON',
            'PENNSYLVANIA',
            'RHODE ISLAND',
            'SOUTH CAROLINA',
            'SOUTH DAKOTA',
            'TENNESSEE',
            'TEXAS',
            'UTAH',
            'VERMONT',
            'VIRGINIA',
            'WASHINGTON',
            'WEST VIRGINIA',
            'WISCONSIN',
            'WYOMING',
        }
    ),
    sam_text_prompt='license plate, registration plate, number plate',
    # Crop-class groups routed straight to the secondary segmenter, skipping
    # the primary detector — the reference detector is known weak on
    # motorcycle plates (near-square, off-axis mounting). Must match the
    # class registry's ``group`` values exactly.
    secondary_shape_groups=frozenset(
        {
            'sportbikes',
            'cruisers',
            'touring-adventurebikes',
            'trikes-dirtbikes-motards-scooters-bicycles',
        }
    ),
)

REFERENCE_PROFILES: MappingProxyType[str, DetectionProfile] = MappingProxyType(
    {REFERENCE_LICENSE_PLATE_PROFILE.name: REFERENCE_LICENSE_PLATE_PROFILE}
)


__all__ = ['REFERENCE_LICENSE_PLATE_PROFILE', 'REFERENCE_PROFILES']
