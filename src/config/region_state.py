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

**Exception (work item B2):** ``no_plate_box`` / ``no_plate_visible``
were the last two domain-specific values on the public
wire contract and were renamed to ``no_region_box`` /
``no_region_visible`` — symbol *and* value — to match the generic
vocabulary the rest of this module already uses. That is a breaking
change to the wire contract, coordinated with the frontend consumer
before landing, and a deployment carrying pre-B2 documents needs a
one-off ``update_by_query`` to rewrite those two status strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RegionStatus(str, Enum):
    """Canonical region-status values written by the curation pipeline."""

    PENDING_DETECTION = 'pending_detection'
    PENDING_VERIFICATION = 'pending_verification'
    DETECTED = 'detected'
    VERIFY_REJECTED = 'verify_rejected'
    NO_REGION_BOX = 'no_region_box'
    NO_REGION_VISIBLE = 'no_region_visible'
    DETECTION_FAILED = 'detection_failed'
    # Human says "this detected box is not a valid region". Box + provenance
    # are PRESERVED (unlike NO_REGION_VISIBLE) for FP analysis + detector
    # hard-negative training.
    FALSE_POSITIVE = 'false_positive'


TERMINAL_STATUSES: frozenset[RegionStatus] = frozenset(
    {
        RegionStatus.DETECTED,
        RegionStatus.NO_REGION_VISIBLE,
        RegionStatus.VERIFY_REJECTED,
        RegionStatus.NO_REGION_BOX,
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


@dataclass(frozen=True)
class RegionStatusInfo:
    """What one status means to a reviewer and to the human writers.

    ``role`` groups statuses the way a UI treats them (``pending``,
    ``positive``, ``rejected``, ``absent``, ``false_positive``,
    ``failed``). ``clears_box``: a human write of this status removes the
    region box and score. ``wants_reason``: a human writing it may attach
    a ``region_rejection_reason``.
    """

    label: str
    role: str
    human_writable: bool
    clears_box: bool = False
    wants_reason: bool = False


REGION_STATUS_INFO: dict[RegionStatus, RegionStatusInfo] = {
    RegionStatus.PENDING_DETECTION: RegionStatusInfo('pending detection', 'pending', False),
    RegionStatus.PENDING_VERIFICATION: RegionStatusInfo('pending verification', 'pending', False),
    RegionStatus.DETECTED: RegionStatusInfo('detected (region visible)', 'positive', True),
    RegionStatus.VERIFY_REJECTED: RegionStatusInfo(
        'rejected (bad detection)', 'rejected', True, wants_reason=True
    ),
    RegionStatus.NO_REGION_BOX: RegionStatusInfo('no box found', 'absent', False, clears_box=True),
    RegionStatus.NO_REGION_VISIBLE: RegionStatusInfo(
        'no region visible', 'absent', True, clears_box=True, wants_reason=True
    ),
    RegionStatus.DETECTION_FAILED: RegionStatusInfo('detection failed', 'failed', False),
    RegionStatus.FALSE_POSITIVE: RegionStatusInfo(
        'false positive (box kept)', 'false_positive', True
    ),
}

# A human confirming a region writes this; "there is no region" writes
# REJECT; "the detector's box is wrong, keep it for training" writes FP.
CONFIRM_STATUS = RegionStatus.DETECTED
REJECT_STATUS = RegionStatus.NO_REGION_VISIBLE
FALSE_POSITIVE_STATUS = RegionStatus.FALSE_POSITIVE

# The values an operator may write by hand, derived from REGION_STATUS_INFO.
# The pipeline also writes transient states (pending_*, detection_failed)
# a human never sets directly.
HUMAN_WRITABLE_STATUSES: frozenset[RegionStatus] = frozenset(
    s for s, info in REGION_STATUS_INFO.items() if info.human_writable
)


@dataclass(frozen=True)
class BoxStateInfo:
    """What one per-box ``state`` (W8) means, served on ``box_states``.

    ``tone`` is a semantic meaning, not a CSS color (W8.7, Cropwright
    ask, coordinator-approved 2026-09-26): Cropwright maps tone to its
    own theme palette instead of keeping its own state->color rules.
    """

    label: str
    role: str
    tone: str
    human_writable: bool
    exported: bool
    dashed: bool = False
    dim: bool = False
    badge: str | None = None


BOX_STATE_INFO: dict[str, BoxStateInfo] = {
    'proposed': BoxStateInfo('awaiting verification', 'proposed', 'proposed', True, False),
    'accepted': BoxStateInfo('accepted', 'accepted', 'accepted', True, True),
    'rejected': BoxStateInfo('rejected', 'rejected', 'rejected', True, False, dashed=True),
    'false_positive': BoxStateInfo(
        'false positive', 'false_positive', 'neutral', True, False, dim=True, badge='false pos'
    ),
}


@dataclass(frozen=True)
class BoxStateRoute:
    """Which per-box ``state`` values one region-box write route accepts,
    and what a new box (``box_id: null``) defaults to when the route
    doesn't set one explicitly (W8.7/W8.8). The served list
    (``box_state_routes`` on ``GET {prefix}/regions/statuses``) and the
    routes' own validators both read this table, so they can't drift."""

    route: str
    states: tuple[str, ...]
    new_box_default: str | None


BOX_STATE_ROUTES: tuple[BoxStateRoute, ...] = (
    BoxStateRoute(
        route='PUT /crops/{crop_id}/regions',
        states=tuple(BOX_STATE_INFO),
        new_box_default='accepted',
    ),
    BoxStateRoute(
        route='PUT /crops/batch_regions',
        states=tuple(BOX_STATE_INFO),
        new_box_default='accepted',
    ),
    BoxStateRoute(
        route='PATCH /crops/{crop_id}/regions/{box_id}',
        states=tuple(BOX_STATE_INFO),
        new_box_default=None,
    ),
    BoxStateRoute(
        route='POST /regions/batch_box_state',
        states=tuple(BOX_STATE_INFO),
        new_box_default=None,
    ),
)


def region_status_catalog() -> dict[str, Any]:
    """The lifecycle vocabulary served by ``GET {prefix}/regions/statuses``."""
    return {
        'statuses': [
            {
                'value': status.value,
                'label': info.label,
                'role': info.role,
                'terminal': status in TERMINAL_STATUSES,
                'human_writable': info.human_writable,
                'clears_box': info.clears_box,
                'wants_reason': info.wants_reason,
            }
            for status, info in ((s, REGION_STATUS_INFO[s]) for s in RegionStatus)
        ],
        'confirm_status': CONFIRM_STATUS.value,
        'reject_status': REJECT_STATUS.value,
        'false_positive_status': FALSE_POSITIVE_STATUS.value,
        'box_states': [
            {
                'value': value,
                'label': info.label,
                'role': info.role,
                'tone': info.tone,
                'human_writable': info.human_writable,
                'exported': info.exported,
                'dashed': info.dashed,
                'dim': info.dim,
                'badge': info.badge,
            }
            for value, info in BOX_STATE_INFO.items()
        ],
        'box_state_routes': [
            {'route': r.route, 'states': list(r.states), 'new_box_default': r.new_box_default}
            for r in BOX_STATE_ROUTES
        ],
    }


__all__ = [
    'BOX_STATE_INFO',
    'BOX_STATE_ROUTES',
    'CONFIRM_STATUS',
    'FALSE_POSITIVE_STATUS',
    'HUMAN_WRITABLE_STATUSES',
    'PENDING_STATUSES',
    'REGION_STATUS_INFO',
    'REJECT_STATUS',
    'TERMINAL_STATUSES',
    'BoxStateInfo',
    'BoxStateRoute',
    'RegionStatus',
    'RegionStatusInfo',
    'region_status_catalog',
]
