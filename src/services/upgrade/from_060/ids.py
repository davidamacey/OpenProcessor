"""Deterministic identifiers: a re-upgrade of the same 0.6.0 install gives the same ids (RF-63).

Every row the load creates from a 0.6.0 source gets a UUID derived from what identifies the
source, so loading twice (after ``--reset``, or after a rollback and a fresh upgrade) produces
the same ids and the same counts.
"""

from __future__ import annotations

import uuid


_NAMESPACE = uuid.UUID('5d7b2a9e-0c1f-4c6e-9a3b-7e0e0a600060')


def stable_id(kind: str, *parts: str | int) -> uuid.UUID:
    return uuid.uuid5(_NAMESPACE, '\x1f'.join([kind, *(str(p) for p in parts)]))
