"""Stored ``region_source`` / ``candidate_source`` provenance vocabulary.

Single source of truth for the generic candidate-provenance strings the
detection worker (:mod:`scripts.curation.worker`) writes to a region's
``region_source`` field, and that ``GET {prefix}/regions/vocabulary``
(:mod:`src.services.curation.region_vocabulary`) serves. Lives under
``src/config`` (not ``scripts``) so both the worker and the API router can
import it without a scripts -> src layering violation.
"""

from __future__ import annotations


CANDIDATE_SEGMENTER = 'segmenter'
CANDIDATE_SEGMENTER_TEXT_HINT = 'segmenter_text_hint'
CANDIDATE_DETECTOR = 'detector'
CANDIDATE_DETECTOR_EXISTING = 'detector_existing'

# Every value the worker can write to a region's ``region_source`` field.
CANDIDATE_SOURCES: tuple[str, ...] = (
    CANDIDATE_SEGMENTER,
    CANDIDATE_SEGMENTER_TEXT_HINT,
    CANDIDATE_DETECTOR,
    CANDIDATE_DETECTOR_EXISTING,
)

__all__ = [
    'CANDIDATE_DETECTOR',
    'CANDIDATE_DETECTOR_EXISTING',
    'CANDIDATE_SEGMENTER',
    'CANDIDATE_SEGMENTER_TEXT_HINT',
    'CANDIDATE_SOURCES',
]
