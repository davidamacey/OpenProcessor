"""Embedding breakdown for the dataset stats rollup.

``embedded`` counts items that have a vector (the authoritative
``exists`` test); ``by_state`` is the recorded reason breakdown.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.services.curation.embedding_state import (
    DEFERRED,
    EMBEDDED,
    FAILED,
    NOT_SELECTED,
    embedded_clause,
    legacy_embedded_clause,
    state_clause,
)


_KNOWN_STATES = (EMBEDDED, NOT_SELECTED, DEFERRED, FAILED)


class EmbeddingByState(BaseModel):
    """Items per recorded ``embedding_state``; every key is always present."""

    embedded: int = 0
    not_selected: int = 0
    deferred: int = 0
    failed: int = 0
    unknown: int = Field(
        default=0,
        description=(
            'Items written before embedding_state existed that have no vector (no recorded '
            'state; not a value of the filter). A legacy item with a vector counts as embedded.'
        ),
    )


class EmbeddingBreakdown(BaseModel):
    embedded: int
    not_embedded: int
    by_state: EmbeddingByState


def embedding_aggregations() -> dict[str, Any]:
    return {
        # One filter per known state: a terms aggregation on embedding_state
        # 400s on indexes whose mapping predates the keyword type.
        'embedding_states': {'filters': {'filters': {s: state_clause(s) for s in _KNOWN_STATES}}},
        'embedded_items': {'filter': embedded_clause()},
        # An item written before embedding_state existed that has its vector is
        # embedded; it must not read as "unknown" beside embedded=N.
        'legacy_embedded_items': {'filter': legacy_embedded_clause()},
    }


def embedding_summary(aggs: dict[str, Any], total: int) -> dict[str, Any]:
    embedded = int((aggs.get('embedded_items') or {}).get('doc_count', 0))
    buckets = (aggs.get('embedding_states') or {}).get('buckets') or {}
    counts = {s: int((buckets.get(s) or {}).get('doc_count', 0)) for s in _KNOWN_STATES}
    # A vector with no recorded state is embedded; a state outside the
    # vocabulary, or none at all, is unknown.
    counts[EMBEDDED] += int((aggs.get('legacy_embedded_items') or {}).get('doc_count', 0))
    counts['unknown'] = max(0, total - sum(counts.values()))
    by_state = EmbeddingByState(**counts)
    return {
        'embedded': embedded,
        'not_embedded': max(0, total - embedded),
        'by_state': by_state.model_dump(),
    }
