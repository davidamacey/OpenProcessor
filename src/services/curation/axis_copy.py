"""Served per-axis label/blurb for ``GET /curation/methods`` ``axes[]``
(W2, any_domain_plan.md §7.6 item 7): the client deletes its local copy
of these strings and renders whatever the endpoint serves.

Split out of :mod:`strategy_registry` (which grew past the pre-commit
700-LOC ratchet once this landed) -- pure static copy, no registry
logic.
"""

from __future__ import annotations

from typing import Any


def axis_copy() -> list[dict[str, str]]:
    """One entry per axis the registry builds strategy entries for."""
    return [
        {
            'axis': 'cluster',
            'label': 'Cluster method',
            'description': 'How crops are grouped into clusters for review.',
        },
        {
            'axis': 'score',
            'label': 'Crop scores',
            'description': 'Additive per-crop signals (mistakenness, uniqueness, near-dup).',
        },
        {
            'axis': 'sort',
            'label': 'Sort order',
            'description': 'The order review queues surface items in.',
        },
        {
            'axis': 'overlay',
            'label': 'Overlay',
            'description': 'Additive visual overlays on the review canvas.',
        },
        {'axis': 'export', 'label': 'Export', 'description': 'Dataset export kind.'},
        {
            'axis': 'detection_profile',
            'label': 'Region profile',
            'description': (
                'The active region-detection profile. Settable: activating '
                'a profile here switches the detection worker onto it (the '
                'latest saved revision) without a restart.'
            ),
        },
        {
            'axis': 'prompt_pack',
            'label': 'Prompt pack',
            'description': (
                'The active VLM prompt pack. Settable: activating a pack '
                'here switches VLM calls onto it (the latest saved '
                'revision) without a restart.'
            ),
        },
    ]


def detection_profile_strategies(default_id: str | None) -> list[dict[str, Any]]:
    """Configured sub-region ``DetectionProfile`` axis.

    Reads :mod:`src.services.detection.profile_registry` -- a real,
    process-lifetime registry a deployment can add more than one profile
    to (e.g. a badge profile AND a shipping-label profile) --
    rather than hardcoding any one profile here. Neutral by default: an
    unconfigured deployment registers nothing, so this axis is empty; the
    env-selected profile (``OP_REGION_PROFILE`` / ``OP_REGION_DETECTION_*``)
    plus anything startup code registers is listed.

    ``default_id`` is :func:`resolve_effective_default`'s answer for the
    ``'detection_profile'`` axis (falls back to
    ``get_default_profile_name()`` with no shared-settings override)."""
    # Import triggers cascade_detect's module-level env resolution if it
    # hasn't run yet in this process.
    from src.services.detection import cascade_detect  # noqa: F401
    from src.services.detection.profile_registry import get_profiles

    profiles = get_profiles()
    entries = [
        {
            'id': profile.name,
            'axis': 'detection_profile',
            'label': profile.name,
            'status': 'stable',
            'default': profile.name == default_id,
        }
        for profile in profiles.values()
    ]
    if profiles:
        # W2: an explicit "no active profile" choice, distinct from the
        # unconfigured-deployment empty axis. Only offered once there is
        # at least one profile to turn off.
        entries.append(
            {
                'id': 'off',
                'axis': 'detection_profile',
                'label': 'off',
                'status': 'stable',
                'default': default_id is None,
            }
        )
    return entries


__all__ = ['axis_copy', 'detection_profile_strategies']
