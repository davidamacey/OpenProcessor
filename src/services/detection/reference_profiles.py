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
    detector_model='lpr_nanov11_640',
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
