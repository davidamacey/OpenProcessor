"""Per-queue status served on ``GET /review/tabs``: which queues exist and how many items each holds.

A queue that cannot fill is absent from the list; ``absent_queues`` gives the sentence the next steps
(``workflow_steps``) serve for it. The counts follow the caller's filter, in one ``_msearch``
(https://docs.opensearch.org/latest/api-reference/multi-search/) of size-0 searches with exact
totals, so the rail needs no request per queue.
"""

from __future__ import annotations

from typing import Any

from src.clients.curation_opensearch.msearch import msearch_checked
from src.config.curation_accessors import items_index
from src.services.curation.image_join import resolve_filter_join
from src.services.curation.review_filters import ReviewFilters
from src.services.curation.review_request import review_query
from src.services.embedding_spaces.readers import item_write_fields


NO_AUDIT_SAMPLE = (
    'No audit sample is pending. Draw one to check how often the machine labels are right.'
)


def absent_queues(*, has_pending_audit: bool) -> dict[str, str]:
    """Each queue that cannot hold items in this project right now, with why."""
    return {} if has_pending_audit else {'audit': NO_AUDIT_SAMPLE}


async def tab_counts(opensearch: Any, tabs: list[str], filters: ReviewFilters) -> dict[str, int]:
    """Items each queue matches under ``filters``, from one ``_msearch``; a failed entry fails
    the call."""
    if not tabs:
        return {}
    embedding = await item_write_fields()
    join = await resolve_filter_join(opensearch, filters.item)
    bodies = [
        {
            'size': 0,
            'track_total_hits': True,
            'query': review_query(tab, filters, embedding, join)[0],
        }
        for tab in tabs
    ]
    responses = await msearch_checked(opensearch, items_index(), bodies)
    return {
        tab: int(((entry.get('hits') or {}).get('total') or {}).get('value', 0))
        for tab, entry in zip(tabs, responses, strict=True)
    }
