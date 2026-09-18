"""Curation router sub-module — pipeline health snapshot helper.

Split out of :mod:`pipeline` so the parent module stays under the
700-LOC pre-commit ceiling. Exposes exactly one public helper:

* :func:`pipeline_health_snapshot` — one OpenSearch round-trip that
  rolls up the counts the labeler dashboard cares about (validated,
  unvalidated, has_class, cluster_id_mismatched).

Used by ``pipeline_auto_label`` to render the ``baseline`` and
``after`` blocks on the run's final result so the labeler can show a
real before/after delta.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.logging import get_logger
from src.routers.curation._common import CURATION_ITEMS_INDEX


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)


async def pipeline_health_snapshot(opensearch: AsyncOpenSearch) -> dict[str, int]:
    """Roll up the counts a curator cares about into one OS round-trip.

    Returns ``{}`` on any OpenSearch error so the caller can degrade
    gracefully (the snapshot is observability, never load-bearing).
    """
    try:
        resp = await opensearch.search(
            index=CURATION_ITEMS_INDEX,
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
                    'mismatched_cluster': {
                        'filter': {
                            'bool': {
                                'must': [{'exists': {'field': 'class_id'}}],
                                'must_not': [
                                    {
                                        'script': {
                                            'script': {
                                                'source': (
                                                    "doc['cluster_id'].size() > 0 && "
                                                    "doc['class_id'].size() > 0 && "
                                                    "doc['cluster_id'].value == "
                                                    "doc['class_id'].value"
                                                ),
                                                'lang': 'painless',
                                            }
                                        }
                                    }
                                ],
                            }
                        }
                    },
                },
            },
        )
    except Exception as exc:
        logger.warning('pipeline_snapshot_failed', error=str(exc))
        return {}
    aggs = resp.get('aggregations') or {}
    total = ((resp.get('hits') or {}).get('total') or {}).get('value', 0)
    return {
        'total_crops': int(total),
        'validated': int((aggs.get('validated') or {}).get('doc_count', 0)),
        'unvalidated': int((aggs.get('unvalidated') or {}).get('doc_count', 0)),
        'has_class': int((aggs.get('has_class') or {}).get('doc_count', 0)),
        'cluster_id_mismatched': int((aggs.get('mismatched_cluster') or {}).get('doc_count', 0)),
    }
