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

from src.services.curation.ingest_class_sources import classifier_class_sources
from src.services.curation.vlm_class_attempt import recent_empty_answer_clause


COMBINED_RECENT_WINDOW = timedelta(hours=24)


def vlm_selection_query(
    *,
    class_id: int | None,
    cluster_id: int | None,
    classifier_confidence_skip_vlm: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The ``query`` for the VLM stage's scroll."""
    filters: list[dict[str, Any]] = []
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
    query: dict[str, Any] = {'bool': {'must_not': must_not}}
    if filters:
        query['bool']['filter'] = filters
    return query


def unvalidated_count_query(*, class_id: int | None, cluster_id: int | None) -> dict[str, Any]:
    """Unvalidated items left in the run's scope (the dashboard's count)."""
    filters = [
        {'term': {field: value}}
        for field, value in (('class_id', class_id), ('cluster_id', cluster_id))
        if value is not None
    ]
    bool_q: dict[str, Any] = {'must_not': [{'term': {'class_validated': True}}]}
    if filters:
        bool_q['filter'] = filters
    return {'bool': bool_q}


__all__ = ['COMBINED_RECENT_WINDOW', 'unvalidated_count_query', 'vlm_selection_query']
