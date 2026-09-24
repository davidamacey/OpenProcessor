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
from src.services.curation.cluster_ids import cluster_kind


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


def unexclusion_update(
    current: dict[str, Any],
    *,
    now: str,
    live_candidate_ids: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """Update doc reversing an exclusion; ``{}`` if ``current`` isn't excluded.

    A validated item returns to its class cluster (``cluster_id =
    class_id``), keeping its sub-cluster only if it was recorded in that
    same cluster. An unvalidated item excluded from its own class cluster
    (``prior == class_id``) returns there too. An unvalidated item
    excluded from a candidate cluster
    that still has members (``live_candidate_ids``, resolved by the
    caller) returns to it. Anything else drops to the residual pool
    (``cluster_id=None``) for a fresh candidate assignment on the next
    recluster -- a candidate id with no members left may have been
    renumbered, so it is not trusted.

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
    prior = current.get(PRIOR_CLUSTER_ID)
    if validated or (class_id is not None and cluster_kind(prior) == 'class' and prior == class_id):
        update['cluster_id'] = class_id
        if prior == class_id:
            update['cluster_subid'] = current.get(PRIOR_CLUSTER_SUBID)
    elif cluster_kind(prior) == 'candidate' and prior in live_candidate_ids:
        update['cluster_id'] = prior
        update['cluster_subid'] = current.get(PRIOR_CLUSTER_SUBID)
    return update


async def live_candidate_ids(opensearch: Any, index: str, crop_ids: list[str]) -> frozenset[int]:
    """Candidate clusters these excluded crops came from that still have members."""
    resp = await opensearch.mget(index=index, body={'ids': list(crop_ids)})
    docs = [d.get('_source') or {} for d in resp.get('docs') or [] if d.get('found')]
    live: set[int] = set()
    for cid in prior_candidate_ids(docs):
        if await cluster_member_count(opensearch, index, cid) > 0:
            live.add(cid)
    return frozenset(live)


async def cluster_member_count(opensearch: Any, index: str, cluster_id: int) -> int:
    """Items currently in ``cluster_id``."""
    n = await opensearch.count(index=index, body={'query': {'term': {'cluster_id': cluster_id}}})
    return int((n or {}).get('count', 0))


def prior_candidate_ids(docs: list[dict[str, Any]]) -> set[int]:
    """Candidate cluster ids the excluded ``docs`` were taken out of."""
    return {
        doc[PRIOR_CLUSTER_ID]
        for doc in docs
        if doc.get('class_excluded') and cluster_kind(doc.get(PRIOR_CLUSTER_ID)) == 'candidate'
    }


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
    'cluster_member_count',
    'exclusion_update',
    'live_candidate_ids',
    'park_restored_state_while_excluded',
    'prior_candidate_ids',
    'unexclusion_update',
]
