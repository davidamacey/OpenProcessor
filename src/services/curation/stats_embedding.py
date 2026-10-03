"""Embedding breakdown for the dataset stats rollup.

``embedded`` counts items that have a vector (the authoritative
``exists`` test); ``by_state`` is the recorded reason breakdown.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.services.curation.embedding_state import embedded_clause


UNKNOWN_STATE_BUCKET = '__none__'


class EmbeddingByState(BaseModel):
    """Items per recorded ``embedding_state``; every key is always present."""

    embedded: int = 0
    not_selected: int = 0
    deferred: int = 0
    failed: int = 0
    unknown: int = Field(
        default=0,
        description='Items written before embedding_state existed (no recorded state; not a value of the filter).',
    )


class EmbeddingBreakdown(BaseModel):
    embedded: int
    not_embedded: int
    by_state: EmbeddingByState


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
    counts: dict[str, int] = {}
    for b in buckets:
        key = str(b['key'])
        # A state outside the vocabulary counts as unknown rather than vanishing.
        if key == UNKNOWN_STATE_BUCKET or key not in EmbeddingByState.model_fields:
            key = 'unknown'
        counts[key] = counts.get(key, 0) + int(b['doc_count'])
    by_state = EmbeddingByState(**counts)
    return {
        'embedded': embedded,
        'not_embedded': max(0, total - embedded),
        'by_state': by_state.model_dump(),
    }
