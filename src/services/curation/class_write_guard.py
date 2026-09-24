"""Write-time re-check for automated class writers (VLM / auto-label / worker).

An automated class writer reads an item, spends seconds to minutes deciding
(a VLM round trip, a cluster vote), then writes. The OCC write only proves
nothing changed between the *write-time* re-read and the write; without
more, a human write that landed during the decision — an undo restoring the
classifier label, a relabel — is silently overwritten by a decision made
on state that no longer exists. (Live: a VLM label replaced a classifier
label a human undo had restored 17 s earlier.)

:class:`ClassWriteGuard` closes that window: the writer records a token of
the class state it decided on (:func:`class_state_token`) at read time, and
its OCC merger lets the write through only if the write-time state still
has the same token and is not human-owned/validated
(:func:`class_write_locked`). Combined with the conditional write
(``if_seq_no`` / ``if_primary_term``) that every merger runs under, the
write lands only on exactly the class state the decision was made on.

The token includes the class-history length and last entry, so a
round trip (label then undo back to the same class) still counts as a
change: a human touched the item after the read.
"""

from __future__ import annotations

from typing import Any

from src.clients.occ import is_human_owned_class
from src.core.logging import get_logger


logger = get_logger(__name__)

CLASS_GUARD_SOURCE_FIELDS: tuple[str, ...] = (
    'class_id',
    'class_name',
    'class_source',
    'label_source',
    'class_validated',
    'class_labeled_at',
    'class_excluded',
    'class_id_history',
)
"""``_source`` fields a reader must fetch for :func:`class_state_token`."""


def class_state_token(source: dict[str, Any]) -> tuple[Any, ...]:
    """A comparable fingerprint of an item's class state."""
    history = source.get('class_id_history') or []
    last = history[-1] if history and isinstance(history[-1], dict) else {}
    return (
        source.get('class_id'),
        source.get('class_name'),
        source.get('class_source'),
        source.get('label_source'),
        bool(source.get('class_validated')),
        source.get('class_labeled_at'),
        bool(source.get('class_excluded')),
        len(history),
        last.get('writer'),
        last.get('at'),
    )


def class_write_locked(source: dict[str, Any]) -> bool:
    """True when no automated class write may touch the item: a human owns
    its class, or any writer validated it."""
    return is_human_owned_class(source) or bool(source.get('class_validated'))


class ClassWriteGuard:
    """Per-run record of the class state each item was read in."""

    def __init__(self, writer_id: str) -> None:
        self.writer_id = writer_id
        self._tokens: dict[str, tuple[Any, ...]] = {}
        self.stale: list[str] = []

    def remember(self, doc_id: str, source: dict[str, Any]) -> None:
        self._tokens[doc_id] = class_state_token(source)

    def token(self, doc_id: str) -> tuple[Any, ...] | None:
        return self._tokens.get(doc_id)

    def allows(self, doc_id: str, current: dict[str, Any]) -> bool:
        """Whether the write-time doc ``current`` may take this writer's
        class write. Never true for an item this run did not read."""
        if class_write_locked(current):
            return False
        read = self._tokens.get(doc_id)
        if read is not None and class_state_token(current) == read:
            return True
        self.stale.append(doc_id)
        logger.info('class_write_stale_skip', doc_id=doc_id, writer_id=self.writer_id)
        return False


def class_write_allowed(read_token: tuple[Any, ...] | None, current: dict[str, Any]) -> bool:
    """Single-item form of :meth:`ClassWriteGuard.allows` for writers that
    carry the read token on their own task objects."""
    if class_write_locked(current) or read_token is None:
        return False
    return class_state_token(current) == read_token


__all__ = [
    'CLASS_GUARD_SOURCE_FIELDS',
    'ClassWriteGuard',
    'class_state_token',
    'class_write_allowed',
    'class_write_locked',
]
