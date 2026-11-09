"""Whether an item has an embedding vector, and why not when it does not.

The state (a keyword on the items index, one field per embedding space) records the REASON an
item has no vector in that space; the authoritative test for "has a vector" is always the
``exists`` query below. A document written before the field existed has no state (unknown), not
``embedded``.

Every consumer that needs "embedded" or "not embedded" builds its clause here, from the vector
and state fields the registry names for the space it reads
(``src.services.embedding_spaces.readers``); nothing else names an ``exists`` clause on an item
vector field (``tests/curation/test_embedding_state_consumers.py`` pins that).
"""

from __future__ import annotations

from typing import Any, Literal


EMBEDDED = 'embedded'
# Ingest policy mode ``selected``: this detection was left out of the embedding.
NOT_SELECTED = 'not_selected'
# No vector yet by design: ingest policy mode ``lazy`` stores no vector at ingest, and a
# vector dropped because the target project could not use it is stored this way too.
DEFERRED = 'deferred'
# The encoder raised for this item at ingest.
FAILED = 'failed'

EmbeddingState = Literal['embedded', 'not_selected', 'deferred', 'failed']


def embedded_clause(vector_field: str) -> dict[str, Any]:
    """Items that have the vector ``vector_field``."""
    return {'exists': {'field': vector_field}}


def state_clause(state: str, state_field: str) -> dict[str, Any]:
    """Items whose recorded state (in ``state_field``) is ``state``.

    ``match_phrase`` answers identically on a keyword mapping and on the
    dynamic ``text`` mapping an index created before the field was mapped
    carries (a ``term`` or a terms aggregation would 400 or miss on those).
    """
    return {'match_phrase': {state_field: state}}


def not_embedded_clause(vector_field: str) -> dict[str, Any]:
    """Items without the vector ``vector_field``."""
    return {'bool': {'must_not': [embedded_clause(vector_field)]}}


def legacy_embedded_clause(vector_field: str, state_field: str) -> dict[str, Any]:
    """Items that have the vector but were written before the state field
    existed: embedded by the authoritative test, with no recorded state."""
    return {
        'bool': {
            'must': [embedded_clause(vector_field)],
            'must_not': [{'exists': {'field': state_field}}],
        }
    }


def legacy_embedded_backfill_body(vector_field: str, state_field: str) -> dict[str, Any]:
    """The ``update_by_query`` body that records ``embedded`` on every legacy item that has its
    vector, so the stored state, the wire item, the stats breakdown and the filter agree (the
    vector is excluded from item reads, so the wire cannot derive it)."""
    return {
        'query': legacy_embedded_clause(vector_field, state_field),
        'script': {
            'lang': 'painless',
            'source': f"ctx._source.{state_field} = '{EMBEDDED}'",
        },
    }


def keep_stored_vector_state(
    merged: dict[str, Any], existing: dict[str, Any], vector_field: str, state_field: str
) -> None:
    """Drop (in place) an incoming state that would contradict a stored vector.

    An update without the vector leaves the stored vector in place, so a re-ingest whose
    encoder call failed must not relabel an ``embedded`` item as ``failed``.
    """
    if vector_field not in merged and existing.get(state_field) == EMBEDDED:
        merged.pop(state_field, None)


__all__ = [
    'DEFERRED',
    'EMBEDDED',
    'FAILED',
    'NOT_SELECTED',
    'EmbeddingState',
    'embedded_clause',
    'keep_stored_vector_state',
    'legacy_embedded_backfill_body',
    'legacy_embedded_clause',
    'not_embedded_clause',
    'state_clause',
]
