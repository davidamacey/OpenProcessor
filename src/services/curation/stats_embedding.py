"""Embedding breakdown for the dataset stats rollup.

``embedded`` counts items that have a vector (the authoritative
``exists`` test); ``by_state`` is the recorded reason breakdown, with
``unknown`` for documents written before ``embedding_state`` existed.
"""

from __future__ import annotations

from typing import Any

from src.services.curation.embedding_state import embedded_clause


UNKNOWN_STATE_BUCKET = '__none__'


def embedding_aggregations() -> dict[str, Any]:
    return {
        'embedding_states': {
            'terms': {'field': 'embedding_state', 'size': 16, 'missing': UNKNOWN_STATE_BUCKET}
        },
        'embedded_items': {'filter': embedded_clause()},
    }


def embedding_summary(aggs: dict[str, Any], total: int) -> dict[str, Any]:
    embedded = int((aggs.get('embedded_items') or {}).get('doc_count', 0))
    buckets = (aggs.get('embedding_states') or {}).get('buckets') or []
    by_state = {
        ('unknown' if b['key'] == UNKNOWN_STATE_BUCKET else str(b['key'])): int(b['doc_count'])
        for b in buckets
    }
    return {'embedded': embedded, 'not_embedded': max(0, total - embedded), 'by_state': by_state}
