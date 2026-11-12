"""A real-state reason for a zero-result ``GET /review/{tab}``.

Split out of :mod:`review_queries` (which owns tab *selection*) to keep
that module under the 700-LOC pre-commit ceiling -- this module owns
*why a queue came back empty*, computed from live index state rather than
a canned string, so the labeler can tell an operator what to actually do
instead of a bare "Queue empty."
"""

from __future__ import annotations

from typing import Any

from src.config import IndexRole, get_curation_config, index_name
from src.domain.embedding_state import not_embedded_clause
from src.services.curation.detections_summary import suggested_embed_request
from src.services.curation.item_filter import ItemFilter
from src.services.embedding_spaces.readers import item_vector_field


def _items_index() -> str:
    return index_name(get_curation_config(), IndexRole.ITEMS)


async def _field_has_any_value(opensearch: Any, field: str) -> bool:
    resp = await opensearch.count(
        index=_items_index(), body={'query': {'exists': {'field': field}}}
    )
    return int(resp.get('count', 0)) > 0


async def _count_unembedded(opensearch: Any) -> int:
    vector_field = await item_vector_field()
    resp = await opensearch.count(
        index=_items_index(), body={'query': not_embedded_clause(vector_field)}
    )
    return int(resp.get('count', 0))


REGION_PROFILE_OFF_REASON = (
    'the region profile is off for this project: activate one to review regions'
)


def region_queue_is_off(tab: str) -> bool:
    """The ``regions`` tab has nothing to serve while no region profile is
    active: rows written under an earlier profile are stale, not a queue."""
    from src.services.detection.profile_registry import get_active_region_profile

    return tab == 'regions' and get_active_region_profile() is None


async def compute_empty_reason(tab: str, filters: Any, opensearch: Any) -> str:
    """A real-state reason for a zero-result ``GET /review/{queue}``:

    - the caller asked for a ``min_mistakenness`` floor but no item has
      ever been scored -> "item scores never computed";
    - ``new_class_proposals`` with nothing pending -> "no unclassified
      proposals";
    - otherwise -> "no items match".
    """
    if getattr(filters, 'min_mistakenness', None) is not None and not await _field_has_any_value(
        opensearch, 'mistakenness_score'
    ):
        return 'item scores never computed'
    if tab == 'new_class_proposals':
        return 'no unclassified proposals'
    return 'no items match'


async def review_tabs_empty_state(opensearch: Any) -> dict[str, Any]:
    """The underlying-state flags :func:`compute_empty_reason` reads,
    served once on ``GET /review/tabs`` so a client can annotate ANY
    tab's zero-result state without one ``count`` round-trip per tab.
    ``suggested_reprocess`` (the shared embed request) is set exactly when
    ``has_unembedded_items``, so an empty queue can offer the embed action."""
    has_probe_predictions = await _field_has_any_value(opensearch, 'probe_pred_entropy')
    has_item_scores = await _field_has_any_value(opensearch, 'mistakenness_score')
    unembedded = await _count_unembedded(opensearch)
    return {
        'has_probe_predictions': has_probe_predictions,
        'has_item_scores': has_item_scores,
        'has_unembedded_items': unembedded > 0,
        'suggested_reprocess': suggested_embed_request(ItemFilter()) if unembedded else None,
    }


__all__ = [
    'REGION_PROFILE_OFF_REASON',
    'compute_empty_reason',
    'region_queue_is_off',
    'review_tabs_empty_state',
]
