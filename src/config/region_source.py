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
# A region box written by dataset import (W10) -- stored on RegionBox.source,
# distinct from region_source's worker-candidate values above (which never
# reach a stored box's ``source`` field; that field's only other value is
# the human-created-box marker ``'human'``). Locked like a human box
# (src.clients.occ.is_locked_box) and served in GET /regions/vocabulary's
# box_sources list ("Imported").
CANDIDATE_IMPORT = 'import'

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
    'CANDIDATE_IMPORT',
    'CANDIDATE_SEGMENTER',
    'CANDIDATE_SEGMENTER_TEXT_HINT',
    'CANDIDATE_SOURCES',
]
