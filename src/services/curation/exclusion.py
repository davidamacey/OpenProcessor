"""Exclude / un-exclude update builders for items (the labeler's Ignore + Undo).

Excluding parks an item in the excluded sentinel cluster and drops its
class validation (an excluded item is never a training label). Before
this module, that was a one-way loss: un-exclude could not know whether
the item had been validated, so every un-excluded item — validated class
labels included — fell into the residual pool (``cluster_id=None``),
breaking the ``cluster_id == class_id`` invariant for validated items
until the next recluster.

Exclude now records the pre-exclusion validation and cluster placement in
``excluded_prior_*`` fields; un-exclude restores them.
"""

from __future__ import annotations

from typing import Any

from src.clients.occ import is_human_owned_class


EXCLUDED_CLUSTER_ID = -2
"""Sentinel cluster for human-excluded items (see the clustering
orchestrator's parked-id ranges)."""

PRIOR_VALIDATED = 'excluded_prior_class_validated'
PRIOR_CLUSTER_ID = 'excluded_prior_cluster_id'
PRIOR_CLUSTER_SUBID = 'excluded_prior_cluster_subid'


def exclusion_update(current: dict[str, Any], *, reason: str, now: str) -> dict[str, Any]:
    """Update doc excluding ``current``.

    Re-excluding an already-excluded item only refreshes the reason and
    timestamp: its recorded pre-exclusion state must survive, since the
    live fields now hold the excluded placeholders.
    """
    update: dict[str, Any] = {
        'class_excluded': True,
        'excluded_at': now,
        'excluded_by': 'human',
        'excluded_reason': reason,
        'updated_at': now,
    }
    if current.get('class_excluded'):
        return update
    update.update(
        {
            PRIOR_VALIDATED: bool(current.get('class_validated')),
            PRIOR_CLUSTER_ID: current.get('cluster_id'),
            PRIOR_CLUSTER_SUBID: current.get('cluster_subid'),
            # Leave the candidate bucket so cluster counts drop
            # immediately even before the next recluster.
            'cluster_id': EXCLUDED_CLUSTER_ID,
            'cluster_subid': None,
            # An excluded item is not a validated class label.
            'class_validated': False,
        }
    )
    return update


def unexclusion_update(current: dict[str, Any], *, now: str) -> dict[str, Any]:
    """Update doc reversing an exclusion; ``{}`` if ``current`` isn't excluded.

    A validated item returns to its class cluster (``cluster_id =
    class_id``), keeping its sub-cluster only if it was recorded in that
    same cluster. Anything else drops to the residual pool
    (``cluster_id=None``) for a fresh candidate assignment on the next
    recluster.

    Items excluded before ``excluded_prior_class_validated`` was recorded
    lost their validation flag at exclude time; for those a human-sourced
    class (the only validated class a human exclusion could have wiped
    without a record) is taken as validated.
    """
    if not current.get('class_excluded'):
        return {}
    if PRIOR_VALIDATED in current and current[PRIOR_VALIDATED] is not None:
        validated = bool(current[PRIOR_VALIDATED])
    else:
        validated = is_human_owned_class(current)
    class_id = current.get('class_id')
    validated = validated and class_id is not None

    update: dict[str, Any] = {
        'class_excluded': False,
        'excluded_at': None,
        'excluded_by': None,
        'excluded_reason': None,
        PRIOR_VALIDATED: None,
        PRIOR_CLUSTER_ID: None,
        PRIOR_CLUSTER_SUBID: None,
        'class_validated': validated,
        'cluster_id': None,
        'cluster_subid': None,
        'updated_at': now,
    }
    if validated:
        update['cluster_id'] = class_id
        if current.get(PRIOR_CLUSTER_ID) == class_id:
            update['cluster_subid'] = current.get(PRIOR_CLUSTER_SUBID)
    return update


def park_restored_state_while_excluded(restored: dict[str, Any]) -> dict[str, Any]:
    """Redirect a class-state restore (label Undo) on an excluded item.

    The item stays excluded, so its live validation/cluster fields keep
    their excluded placeholders; the restored values become the
    ``excluded_prior_*`` state that un-exclude will apply.
    """
    out = dict(restored)
    out[PRIOR_VALIDATED] = bool(out.pop('class_validated'))
    out[PRIOR_CLUSTER_ID] = out.pop('cluster_id')
    out[PRIOR_CLUSTER_SUBID] = out.pop('cluster_subid')
    return out


__all__ = [
    'EXCLUDED_CLUSTER_ID',
    'PRIOR_CLUSTER_ID',
    'PRIOR_CLUSTER_SUBID',
    'PRIOR_VALIDATED',
    'exclusion_update',
    'park_restored_state_while_excluded',
    'unexclusion_update',
]
