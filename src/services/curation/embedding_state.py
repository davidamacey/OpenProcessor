"""Whether an item has an embedding vector, and why not when it does not.

``embedding_state`` (a keyword on the items index) records the REASON an item
has no ``pe_embedding``; the authoritative test for "has a vector" is always
the ``exists`` query below. A document written before the field existed has no
state (unknown), not ``embedded``.

Every consumer that needs "embedded" or "not embedded" builds its clause here;
nothing else names an ``exists`` clause on the item embedding field
(``tests/curation/test_embedding_state_consumers.py`` pins that).
"""

from __future__ import annotations

from typing import Any, Literal

from src.config.curation import ITEM_EMBEDDING_FIELD


EMBEDDED = 'embedded'
# Reserved for the selective-embedding policy; written by no path yet.
NOT_SELECTED = 'not_selected'
# No vector yet by design: the vector was dropped because the target project
# could not use it.
DEFERRED = 'deferred'
# The encoder raised for this item at ingest.
FAILED = 'failed'

EmbeddingState = Literal['embedded', 'not_selected', 'deferred', 'failed']


def embedded_clause(field: str = ITEM_EMBEDDING_FIELD) -> dict[str, Any]:
    """Items that have the vector ``field`` (default: the item embedding)."""
    return {'exists': {'field': field}}


def not_embedded_clause(field: str = ITEM_EMBEDDING_FIELD) -> dict[str, Any]:
    """Items without the vector ``field``."""
    return {'bool': {'must_not': [embedded_clause(field)]}}


def keep_stored_vector_state(merged: dict[str, Any], existing: dict[str, Any]) -> None:
    """Drop (in place) an incoming ``embedding_state`` that would contradict a
    stored vector.

    An update without a ``pe_embedding`` leaves the stored vector in place, so
    a re-ingest whose encoder call failed must not relabel an ``embedded``
    item as ``failed``.
    """
    if ITEM_EMBEDDING_FIELD not in merged and existing.get('embedding_state') == EMBEDDED:
        merged.pop('embedding_state', None)


__all__ = [
    'DEFERRED',
    'EMBEDDED',
    'FAILED',
    'NOT_SELECTED',
    'EmbeddingState',
    'embedded_clause',
    'keep_stored_vector_state',
    'not_embedded_clause',
]
