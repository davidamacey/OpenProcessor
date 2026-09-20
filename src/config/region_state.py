"""Canonical region-status enum for the curation region-of-interest pipeline.

This is the single source of truth for the region-status values written
to OpenSearch under the configured items index (see
:class:`src.config.region_fields.RegionFields`). See
``docs/design/curation_design_rationale.md`` for the broader field-
indirection design this complements.

State machine:

    ingest -> pending_detection -> detector cascade ----+
                                    verify+extract       +-> detected
                                                          +-> verify_rejected
                                                          +-> no_region_box
    ingest -> pending_verification -> verify  -----------+
                                                          +-> no_region_visible

    sanity-gate reject (any path) -> detection_failed

A human reviewer can also mark a detection a ``false_positive``: the
detector drew a box but it is NOT a valid region. Unlike
``no_region_visible`` (box deleted, "nothing here"), ``false_positive``
KEEPS the box + all detection provenance so the bad detection can be
analysed and fed back into detector training as a hard negative.

On-disk string values are kept byte-identical to the values already
written by earlier code (``pending_detection``, ``detected``, etc.) —
same no-migration reasoning as ``RegionFields``: this is a rename of
the Python symbol, not the OpenSearch data.
"""

from __future__ import annotations

from enum import Enum


class RegionStatus(str, Enum):
    """Canonical region-status values written by the curation pipeline."""

    PENDING_DETECTION = 'pending_detection'
    PENDING_VERIFICATION = 'pending_verification'
    DETECTED = 'detected'
    VERIFY_REJECTED = 'verify_rejected'
    NO_PLATE_BOX = 'no_plate_box'
    NO_PLATE_VISIBLE = 'no_plate_visible'
    DETECTION_FAILED = 'detection_failed'
    # Human says "this detected box is not a valid region". Box + provenance
    # are PRESERVED (unlike NO_PLATE_VISIBLE) for FP analysis + detector
    # hard-negative training.
    FALSE_POSITIVE = 'false_positive'


TERMINAL_STATUSES: frozenset[RegionStatus] = frozenset(
    {
        RegionStatus.DETECTED,
        RegionStatus.NO_PLATE_VISIBLE,
        RegionStatus.VERIFY_REJECTED,
        RegionStatus.NO_PLATE_BOX,
        RegionStatus.DETECTION_FAILED,
        RegionStatus.FALSE_POSITIVE,
    }
)


PENDING_STATUSES: frozenset[RegionStatus] = frozenset(
    {
        RegionStatus.PENDING_DETECTION,
        RegionStatus.PENDING_VERIFICATION,
    }
)


__all__ = ['PENDING_STATUSES', 'TERMINAL_STATUSES', 'RegionStatus']
