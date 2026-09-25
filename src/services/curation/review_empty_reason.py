"""C3: a real-state reason for a zero-result ``GET /review/{tab}``.

Split out of :mod:`review_queries` (which owns tab *selection*) to keep
that module under the 700-LOC pre-commit ceiling -- this module owns
*why a tab came back empty*, computed from live index state rather than
a canned string, so the labeler can tell an operator what to actually do
("run a probe") instead of a bare "Queue empty."
"""

from __future__ import annotations

from typing import Any

from src.config import IndexRole, get_curation_config, index_name


# Tabs whose own selection requires a probe-scored item
# (review_queries.build_tab_query's 'uncertainty' / 'model_disagreements'
# branches both add an `exists` clause on one of these fields) -- an
# empty result on one of these almost always means no probe pass has run
# yet, not that the pool has nothing uncertain.
_PROBE_GATED_TABS: dict[str, str] = {
    'uncertainty': 'probe_pred_entropy',
    'model_disagreements': 'probe_pred_class',
}


def _items_index() -> str:
    return index_name(get_curation_config(), IndexRole.ITEMS)


async def _field_has_any_value(opensearch: Any, field: str) -> bool:
    resp = await opensearch.count(
        index=_items_index(), body={'query': {'exists': {'field': field}}}
    )
    return int(resp.get('count', 0)) > 0


async def compute_empty_reason(tab: str, filters: Any, opensearch: Any) -> str:
    """A real-state reason for a zero-result ``GET /review/{tab}``:

    - a probe-gated tab (``uncertainty`` / ``model_disagreements``) with
      no probe-scored item at all -> "no probe predictions — run a probe";
    - the caller asked for a ``min_mistakenness`` floor but no item has
      ever been scored -> "item scores never computed";
    - ``new_class_proposals`` with nothing pending -> "no unclassified
      proposals";
    - otherwise -> "no items match".
    """
    probe_field = _PROBE_GATED_TABS.get(tab)
    if probe_field is not None and not await _field_has_any_value(opensearch, probe_field):
        return 'no probe predictions — run a probe'
    if getattr(filters, 'min_mistakenness', None) is not None and not await _field_has_any_value(
        opensearch, 'mistakenness_score'
    ):
        return 'item scores never computed'
    if tab == 'new_class_proposals':
        return 'no unclassified proposals'
    return 'no items match'


async def review_tabs_empty_state(opensearch: Any) -> dict[str, bool]:
    """The underlying-state flags :func:`compute_empty_reason` reads,
    served once on ``GET /review/tabs`` so a client can annotate ANY
    tab's zero-result state without one ``count`` round-trip per tab."""
    return {
        'has_probe_predictions': await _field_has_any_value(opensearch, 'probe_pred_entropy'),
        'has_item_scores': await _field_has_any_value(opensearch, 'mistakenness_score'),
    }


__all__ = ['compute_empty_reason', 'review_tabs_empty_state']
