"""Pipeline health snapshot for the auto-label run. Exposes exactly one public helper:

* :func:`pipeline_health_snapshot` — one OpenSearch round-trip that
  rolls up the counts the labeler dashboard cares about (validated,
  unvalidated, has_class, unembedded).

Used by the auto-label run to render the ``baseline`` and
``after`` blocks on the run's final result so the labeler can show a
real before/after delta.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opensearchpy.exceptions import OpenSearchException

from src.config.curation_accessors import items_index
from src.core.logging import get_logger
from src.domain.embedding_state import not_embedded_clause
from src.services.embedding_spaces.readers import item_vector_field


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)


async def pipeline_health_snapshot(opensearch: AsyncOpenSearch) -> dict[str, int]:
    """Roll up the counts a curator cares about into one OS round-trip.

    Returns ``{}`` on any OpenSearch error so the caller can degrade
    gracefully (the snapshot is observability, never load-bearing).
    """
    vector_field = await item_vector_field()
    try:
        resp = await opensearch.search(
            index=items_index(),
            body={
                'size': 0,
                'track_total_hits': True,
                'aggs': {
                    'validated': {'filter': {'term': {'class_validated': True}}},
                    'unvalidated': {
                        'filter': {
                            'bool': {
                                'must_not': [{'term': {'class_validated': True}}],
                            }
                        }
                    },
                    'has_class': {'filter': {'exists': {'field': 'class_id'}}},
                    # Not yet in the working set: no vector, so the VLM, clustering
                    # and search skip them.
                    'unembedded': {'filter': not_embedded_clause(vector_field)},
                },
            },
        )
    except OpenSearchException as exc:
        logger.warning('pipeline_snapshot_failed', error=str(exc))
        return {}
    aggs = resp.get('aggregations') or {}
    total = ((resp.get('hits') or {}).get('total') or {}).get('value', 0)
    return {
        'total_crops': int(total),
        'validated': int((aggs.get('validated') or {}).get('doc_count', 0)),
        'unvalidated': int((aggs.get('unvalidated') or {}).get('doc_count', 0)),
        'has_class': int((aggs.get('has_class') or {}).get('doc_count', 0)),
        'unembedded': int((aggs.get('unembedded') or {}).get('doc_count', 0)),
    }
