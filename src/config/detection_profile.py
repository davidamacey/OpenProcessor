"""Detection-cascade heuristics and Triton model wiring, as data.

Replaces hardcoded per-domain constants in the reference license-plate
detection cascade and verification modules (see
``docs/design/oss_genericization_phase2_plan.md`` §3.3 for exact
reference file/line provenance). A ``DetectionProfile`` instance
describes one detectable "region of interest" type (e.g. a license
plate on a vehicle); a deployment with a different region type (a box,
a tractor, …) constructs its own instance instead of forking the
detection-cascade code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DetectionProfile:
    """Aspect/area heuristics and Triton model identity for one region type.

    ``text_pattern`` is a plain ``str``, not a compiled ``re.Pattern`` —
    a frozen dataclass holding a compiled pattern is not cleanly
    serializable and breaks dataclass equality in tests. Compile it once
    at module scope in the consumer instead.
    """

    name: str
    detector_model: str = ''
    detector_version: str = '1'
    input_size: int = 640
    confidence_floor: float = 0.4
    batch_limit: int = 16
    letterbox_fill: tuple[int, int, int] = (114, 114, 114)
    aspect_min: float = 1.2
    aspect_max: float = 8.0
    text_hint_aspect_min: float = 1.5
    text_hint_aspect_max: float = 7.0
    text_hint_rec_floor: float = 0.70
    text_hint_len_min: int = 4
    text_hint_len_max: int = 10
    auto_confirm_aspect: tuple[float, float] = (0.5, 7.0)
    auto_confirm_area_frac: tuple[float, float] = (0.001, 0.40)
    text_pattern: str = r'[A-Z0-9 -]{2,}'

    # Detector identity strings.
    segmenter_name: str = 'sam3'
    segmenter_version: str = '1'
    human_detector_name: str = 'human'
    human_detector_version: str = '1'

    # OCR wiring.
    ocr_det_model: str = 'paddleocr_det_trt'
    ocr_det_version: str = '1'
    ocr_det_input_size: int = 640
    ocr_det_prob_floor: float = 0.30
    ocr_rec_model: str = 'paddleocr_rec'
    ocr_rec_version: str = '1'
    ocr_pipeline_model: str = 'ocr_pipeline'
    sam_text_prompt: str = ''
    secondary_shape_groups: frozenset[str] = field(default_factory=frozenset)
