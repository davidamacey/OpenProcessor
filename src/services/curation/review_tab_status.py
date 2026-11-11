"""Per-tab status served on ``GET /review/tabs``: can the tab fill, why not, how many items.

The counts use the default filters of each tab (test holdout hidden), in one ``_msearch``
(https://docs.opensearch.org/latest/api-reference/multi-search/) of size-0 searches with exact
totals, so the tab rail needs no request per tab.
"""

from __future__ import annotations

from typing import Any

from src.clients.curation_opensearch.msearch import msearch_checked
from src.config.curation_accessors import items_index
from src.services.curation import review_queries
from src.services.curation.review_filter_specs import REGION_STATUS_DEFAULT
from src.services.embedding_spaces.readers import item_vector_field


_PROBE_TABS = frozenset({'uncertainty', 'model_disagreements'})
_NO_PROBE = 'No probe predictions yet. Run a probe from the Dashboard to fill this tab.'
_NO_IMPORTS = 'No imported labels. Import a dataset to fill this tab.'


def unavailable_reason(tab: str, empty_state: dict[str, Any]) -> str | None:
    """Why ``tab`` cannot hold items in this project right now; ``None`` when it can."""
    if tab in _PROBE_TABS and not empty_state['has_probe_predictions']:
        return _NO_PROBE
    if tab == 'imported' and not empty_state['has_imported_labels']:
        return _NO_IMPORTS
    return None


async def tab_counts(opensearch: Any, tabs: list[str]) -> dict[str, int]:
    """Items each tab matches by default, from one ``_msearch``; a failed entry fails the call."""
    if not tabs:
        return {}
    vector_field = await item_vector_field()
    bodies: list[dict[str, Any]] = []
    for tab in tabs:
        must, must_not, _reason = review_queries.build_tab_query(
            tab,
            include_test=False,
            text=None,
            max_rank=None,
            region_status=REGION_STATUS_DEFAULT,
            vector_field=vector_field,
        )
        bodies.append(
            {
                'size': 0,
                'track_total_hits': True,
                'query': {'bool': {'must': must, 'must_not': must_not}},
            }
        )
    responses = await msearch_checked(opensearch, items_index(), bodies)
    return {
        tab: int(((entry.get('hits') or {}).get('total') or {}).get('value', 0))
        for tab, entry in zip(tabs, responses, strict=True)
    }
