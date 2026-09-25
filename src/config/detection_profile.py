"""Detection-cascade heuristics and Triton model wiring, as data.

Replaces hardcoded per-domain constants in the detection cascade and
verification modules (see ``docs/design/curation_design_rationale.md``
§2.3 for the design rationale, including known gaps in how generic the
shipped defaults are today). A ``DetectionProfile`` instance describes
one detectable "region of interest" type (a barcode, a
defect on a manufactured part, a tag on livestock, …); a deployment constructs its own
instance — or loads one from a profile file (see
``examples/region_profiles/``) — instead of forking the
detection-cascade code. No region type ships built in: with no active
profile configured, region detection stays off.
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
    # Region-text reader (src/services/detection/region_text.py). Which
    # reader fills ``region_text``: 'vlm' (the verify call's reading),
    # 'ocr' (the OCR pipeline on the region crop), 'vlm_then_ocr' (OCR
    # when the VLM read nothing), 'both' (store both readings and flag a
    # disagreement). With no VLM configured every mode reads via OCR.
    text_reader: str = 'vlm_then_ocr'
    # Region crop fed to the OCR reader: the region box grown by this
    # fraction of its width/height on each side, clipped to the item crop.
    text_crop_margin: float = 0.05
    # Region crops shorter than this are upscaled to it (and centered on
    # a detector-sized canvas) before OCR; small text otherwise blurs past
    # what the text detector finds.
    text_crop_min_height: int = 56
    # Dominant-text selection: keep lines at least this fraction of the
    # tallest line's height; drop lines centered in this outer band of
    # the crop.
    text_min_height_ratio: float = 0.6
    text_border_margin: float = 0.08
    # Normalization: uppercase, then keep only characters matching the
    # one-character regex ``text_charset`` ('' keeps every non-space
    # character); kept pieces are joined with ``text_join``.
    text_uppercase: bool = False
    text_charset: str = ''
    text_join: str = ' '
    # Accepted reading length (after normalization, spaces not counted);
    # text_len_max=0 means unbounded.
    text_len_min: int = 1
    text_len_max: int = 0
    # Words dropped from every line before selection (compared normalized).
    text_stopwords: frozenset[str] = field(default_factory=frozenset)
    # Minimum recognition score of the weakest kept line.
    text_min_confidence: float = 0.0
    # Reading validity (src/services/detection/region_text_rules.py),
    # applied to every reader's reading before one is chosen. A reading
    # failing a rule is no reading at all. ``text_format``: optional regex
    # the normalized reading must fully match ('' = any).
    # ``text_placeholders``: readings that are never real text (e.g. a
    # prompt's example value; the active prompt pack's quoted examples are
    # added automatically). ``text_reject_sequences``: reject a reading
    # that is one repeated character or one ascending / descending run
    # ("999", "123456", "XYZ") -- stock "I can't read it" answers.
    text_format: str = ''
    text_placeholders: frozenset[str] = field(default_factory=frozenset)
    text_reject_sequences: bool = False
    segmenter_text_prompt: str = ''
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
    # The registry class name this profile's detections should be treated
    # as (e.g. 'defect'), so generic code (class merge guards,
    # training presets, export pairing scans) can special-case "the
    # region class" without hardcoding a domain name. '' (default) means
    # no region class name is configured -- callers must degrade to "not
    # applicable" rather than assume any particular class.
    region_class_name: str = ''
    # Human-readable label for UI surfaces that mention "the region type"
    # (e.g. the review-queue tab title). '' (default) means the caller
    # falls back to a generic label such as "Regions".
    display_name: str = ''
    # The same label for one region (e.g. "Plate" for "Confirm Plate").
    # '' (default) means the caller falls back to "Region".
    display_name_singular: str = ''

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
