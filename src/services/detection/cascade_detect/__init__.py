"""Generic sub-region detection cascade — YOLO-style detector + PaddleOCR.

See ``docs/design/curation_design_rationale.md`` §2.3 / §5.
Wraps a YOLO-style Triton detector to produce sub-region bounding boxes
in the **item crop's** coordinate frame (normalized to ``[0, 1]``), plus
a PaddleOCR-based text detector/recognizer used as a last-resort
rescue path and text-hint source.

Every heuristic (detector identity, confidence floors, aspect bands, OCR
wiring) lives on a :class:`~src.config.DetectionProfile` instance, so a
deployment can describe any region type (a printed label, a box, a
wheel, …) without forking this module. **No profile ships
built in.** ``RegionDetector`` / ``PaddleOcrRegionDetector`` /
``PaddleOcrTextRecognizer`` all require a profile explicitly — the
caller resolves it from :mod:`src.services.detection.profile_registry`
(``get_active_region_profile()``, or a deployment's own registered
profile) or raises, rather than silently falling back to any example
domain. See ``examples/region_profiles/`` for a worked example a
deployment can point ``OP_REGION_PROFILE_PATH`` at.

Modules: ``candidate`` (:class:`RegionCandidate`), ``sanity`` (geometry
guard, provenance, frame transform), ``preprocess`` (decode + letterbox),
``region_detector`` (YOLO decode + :class:`RegionDetector`), ``paddle_det``
(PaddleOCR DBNet rescue detector) and ``ocr_recognizer`` (OCR pipeline
reader). Import symbols from the module that defines them.
"""

from __future__ import annotations

from src.services.detection.profile_registry import ensure_env_region_profile


# Resolve (and register) the deployment's region profile from the
# environment at import time, so a bad OP_REGION_PROFILE name fails loudly
# at startup instead of on the first detection request. Unconfigured is
# valid: no profile is registered and region detection stays off. Any
# import of a submodule runs this first.
ensure_env_region_profile()
