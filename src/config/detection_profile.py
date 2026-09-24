"""Detection-cascade heuristics and Triton model wiring, as data.

Replaces hardcoded per-domain constants in the reference license-plate
detection cascade and verification modules (see
``docs/design/curation_design_rationale.md`` §2.3 for the design
rationale, including known gaps in how generic the shipped defaults
are today). A ``DetectionProfile`` instance
describes one detectable "region of interest" type (e.g. a license
plate on a vehicle); a deployment with a different region type (a box,
a tractor, …) constructs its own instance instead of forking the
detection-cascade code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from typing import Any


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
    # Optional backbone feature-map output of a raw-output (secondary)
    # detector, requested only when Triton reports the model has it.
    # Empty disables the request entirely.
    feature_output: str = 'sppf_feat'
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
    ocr_rec_model: str = 'paddleocr_rec_trt'
    ocr_rec_version: str = '1'
    ocr_pipeline_model: str = 'ocr_pipeline'
    sam_text_prompt: str = ''
    secondary_shape_groups: frozenset[str] = field(default_factory=frozenset)
    # Item (ingest) detectors only: the model class ids whose detections
    # become items. Empty = every class. Lets a generic proposer (e.g. an
    # 80-class COCO model) be narrowed to the classes a deployment curates.
    class_ids: frozenset[int] = field(default_factory=frozenset)
    # Ingest primary only: does this model's class space *be* the class
    # registry? False (the default) = it is a generic proposer (e.g. COCO)
    # whose class ids mean nothing in the registry, so its detections are
    # always unlabeled proposals and a secondary / VLM / human assigns the
    # class. True = confident detections take their class id from the
    # registry directly.
    assigns_class: bool = False
    # Ingest primary only: labels.txt-style file naming the model's own
    # classes (line index = class id), recorded as the proposal name.
    labels_path: str = ''

    @classmethod
    def from_env(
        cls,
        prefix: str,
        *,
        name: str = 'region',
        base: DetectionProfile | None = None,
    ) -> DetectionProfile:
        """Build a :class:`DetectionProfile` from ``{prefix}*`` env vars,
        mirroring :meth:`CurationConfig.from_env` / :meth:`RegionFields.from_env`.

        Prefixes in use: ``OP_INGEST_PRIMARY_`` / ``OP_INGEST_SECONDARY_``
        (ingest item detectors, ``routers/curation/ingest.py``) and
        ``OP_REGION_DETECTION_`` (the region cascade,
        ``services/detection/profile_registry.py``).

        Every field is optional; unset env vars fall back to the
        dataclass default for that field. ``name`` has no dataclass
        default (it is the one required field), so it falls back to the
        ``name`` keyword argument instead when ``{prefix}NAME`` is unset.

        Non-``str`` fields are parsed from their env var's string value
        by inspecting the *default* value's type: ``bool`` from
        ``1/true/yes/on``, ``int``/``float`` via their constructors, a
        ``tuple`` from a comma-separated list (each element converted to
        the tuple's own element type), and a ``frozenset`` from a
        comma-separated list of strings.

        ``base``, when given, supplies the fallback value for every field
        instead of the dataclass defaults (including ``name``) — used to
        layer env overrides on top of a selected named profile.
        """
        defaults = base if base is not None else cls(name=name)
        overrides: dict[str, Any] = {'name': defaults.name}
        for f in fields(defaults):
            raw = os.environ.get(f'{prefix}{f.name.upper()}')
            if raw is None:
                continue
            current = getattr(defaults, f.name)
            if isinstance(current, bool):
                overrides[f.name] = raw.strip().lower() in {'1', 'true', 'yes', 'on'}
            elif isinstance(current, int):
                overrides[f.name] = int(raw)
            elif isinstance(current, float):
                overrides[f.name] = float(raw)
            elif isinstance(current, tuple):
                item_type = type(current[0]) if current else str
                overrides[f.name] = tuple(item_type(part.strip()) for part in raw.split(','))
            elif isinstance(current, frozenset):
                # Element type from the annotation (an empty default carries
                # none): ``frozenset[int]`` fields parse ints.
                elem = int if 'int' in str(f.type) else str
                overrides[f.name] = frozenset(
                    elem(part.strip()) for part in raw.split(',') if part.strip()
                )
            else:
                overrides[f.name] = raw
        return replace(defaults, **overrides)

    @classmethod
    def env_overrides_present(cls, prefix: str) -> bool:
        """``True`` if any ``{prefix}<FIELD>`` env var is set."""
        return any(f'{prefix}{f.name.upper()}' in os.environ for f in fields(cls))


LEGACY_DETECTION_ENV_PREFIX = 'OP_DETECTION_'


def reject_legacy_detection_env() -> None:
    """Fail loudly if any retired ``OP_DETECTION_*`` var is still set.

    ``OP_DETECTION_*`` used to configure the ingest item detector while
    reading like the region detector's config; a deployment setting both
    meanings at once got one silently applied to the other. It is split
    into ``OP_INGEST_PRIMARY_*`` (ingest) and ``OP_REGION_PROFILE`` /
    ``OP_REGION_DETECTION_*`` (region cascade). Leftover vars raise rather
    than being silently ignored or reinterpreted.
    """
    stale = sorted(k for k in os.environ if k.startswith(LEGACY_DETECTION_ENV_PREFIX))
    if stale:
        msg = (
            f'retired env var(s) {stale}: OP_DETECTION_* was split into '
            'OP_INGEST_PRIMARY_* (ingest item detector) and OP_REGION_PROFILE / '
            'OP_REGION_DETECTION_* (region detection) -- rename them'
        )
        raise ValueError(msg)
