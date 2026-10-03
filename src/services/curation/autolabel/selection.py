"""Which items the auto-label VLM stage sends to the VLM.

Two scopes:

* **Global sweep** (default): every unvalidated item except those the VLM
  need not see — classifier-labeled at or above the skip confidence,
  ``vlm_unmatched`` (it already failed once), items the region worker's
  combined call classified in the last 24 h (``vlm_verify_completed_at``),
  and items whose last VLM class attempt in the retry window came back
  empty (``vlm_class_empty_reason``).
  Optionally narrowed to one ``class_id``.
* **Cluster scope** (``cluster_id`` set, ``POST /vlm/label_cluster/{id}``):
  an operator explicitly asked for this cluster, so every unvalidated
  member is labeled — only validated, test-holdout and excluded items are
  left out.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from src.core.logging import get_logger
from src.services.curation.embedding_state import embedded_clause
from src.services.curation.ingest_class_sources import classifier_class_sources
from src.services.curation.item_filter import ItemFilter, item_filter_clauses
from src.services.curation.vlm_class_attempt import recent_empty_answer_clause


logger = get_logger(__name__)

COMBINED_RECENT_WINDOW = timedelta(hours=24)


def vlm_selection_query(
    *,
    class_id: int | None,
    cluster_id: int | None,
    classifier_confidence_skip_vlm: float,
    now: datetime | None = None,
    item_filter: ItemFilter | None = None,
) -> dict[str, Any]:
    """The ``query`` for the VLM stage's scroll."""
    # Only embedded items reach the VLM (the working set), same as the worker.
    filters: list[dict[str, Any]] = [
        embedded_clause(),
        *item_filter_clauses(item_filter or ItemFilter()),
    ]
    if class_id is not None:
        filters.append({'term': {'class_id': class_id}})
    if cluster_id is not None:
        filters.append({'term': {'cluster_id': cluster_id}})
        must_not: list[dict[str, Any]] = [
            {'term': {'class_validated': True}},
            {'term': {'test_holdout': True}},
            {'term': {'class_excluded': True}},
        ]
    else:
        recent_cutoff = ((now or datetime.now(UTC)) - COMBINED_RECENT_WINDOW).isoformat()
        must_not = [
            {'term': {'class_validated': True}},
            {
                'bool': {
                    'filter': [
                        {'terms': {'class_source': sorted(classifier_class_sources())}},
                        {'range': {'confidence': {'gte': classifier_confidence_skip_vlm}}},
                    ],
                },
            },
            {'term': {'class_source': 'vlm_unmatched'}},
            # Classified by the region worker's combined call recently.
            {'range': {'vlm_verify_completed_at': {'gte': recent_cutoff}}},
            # Asked recently and the answer had no class.
            recent_empty_answer_clause(now),
        ]
    return {'bool': {'filter': filters, 'must_not': must_not}}


def unvalidated_count_query(
    *, class_id: int | None, cluster_id: int | None, item_filter: ItemFilter | None = None
) -> dict[str, Any]:
    """Unvalidated items left in the run's scope (the dashboard's count)."""
    filters = [
        {'term': {field: value}}
        for field, value in (('class_id', class_id), ('cluster_id', cluster_id))
        if value is not None
    ]
    filters.extend(item_filter_clauses(item_filter or ItemFilter()))
    bool_q: dict[str, Any] = {'must_not': [{'term': {'class_validated': True}}]}
    if filters:
        bool_q['filter'] = filters
    return {'bool': bool_q}


async def scroll_unvalidated(
    opensearch: Any,
    *,
    index: str,
    query: dict[str, Any],
    source_fields: list[str],
    cap: int | None,
    guard: Any,
) -> list[str]:
    """Every crop id ``query`` matches (up to ``cap``), oldest update first,
    remembering each item's class state in ``guard`` so a later write can tell
    whether it changed."""
    page, ttl = 1000, '5m'
    ids: list[str] = []
    scroll_id: str | None = None
    try:
        resp = await opensearch.search(
            index=index,
            body={
                'size': page,
                '_source': source_fields,
                'query': query,
                'sort': [{'updated_at': 'asc'}],
            },
            scroll=ttl,
        )
        while True:
            scroll_id = resp.get('_scroll_id')
            hits = (resp.get('hits') or {}).get('hits') or []
            if not hits:
                break
            for h in hits:
                crop_id = (h.get('_source') or {}).get('crop_id') or h.get('_id')
                if crop_id:
                    ids.append(crop_id)
                    guard.remember(crop_id, h.get('_source') or {})
                    if cap is not None and len(ids) >= cap:
                        break
            if (cap is not None and len(ids) >= cap) or not scroll_id:
                break
            resp = await opensearch.scroll(scroll_id=scroll_id, scroll=ttl)
    finally:
        if scroll_id:
            try:
                await opensearch.clear_scroll(scroll_id=scroll_id)
            except Exception as exc:
                logger.debug('pipeline_clear_scroll_failed', error=str(exc))
    return ids


__all__ = [
    'COMBINED_RECENT_WINDOW',
    'scroll_unvalidated',
    'unvalidated_count_query',
    'vlm_selection_query',
]
