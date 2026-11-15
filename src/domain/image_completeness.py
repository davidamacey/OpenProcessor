"""When a stored photo counts as stored whole: the images document lists its items.

The ingest writer puts ``item_ids`` (the crop ids of the unit it writes) on the images doc it
creates, before it writes the items. A photo is whole when every listed item exists. The list
only ever names items that exist: whoever removes a listed item takes its id off the list first
(``services/curation/item_delete.py``), so an image whose items were changed on purpose never
looks incomplete.

A document without the field is legacy (written before the field, or copied by a combine): it
is complete, always. ``[]`` is a valid, complete list (a photo with no detections).

An incomplete photo is repaired by the next ingest of the same bytes only when no live writer
can still be writing it: its own writer marked the failure (:data:`INCOMPLETE_MARK_FIELD`), or
the document is older than the repair age. A repair is itself a live writer: it takes a lease
(:data:`REPAIR_LEASE_FIELD`, the time it started) before it writes anything and gives it back
when it ends, so no second repair starts on the image while the lease is younger than the repair
age, marked or not.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


ITEM_IDS_FIELD = 'item_ids'
INCOMPLETE_MARK_FIELD = 'ingest_incomplete'
"""Set by the writer that saw its own items step fail, on the images doc it created."""

REPAIR_LEASE_FIELD = 'repair_started_at'
"""The time a repair of the image started; cleared by the repair that set it when it ends."""

INDEXED_AT_FIELD = 'indexed_at'


STATE_FIELDS = (ITEM_IDS_FIELD, INDEXED_AT_FIELD, INCOMPLETE_MARK_FIELD, REPAIR_LEASE_FIELD)
"""What a reader of an images doc needs to tell whole from incomplete."""


def listed_item_ids(source: dict[str, Any]) -> list[str] | None:
    """The items the images doc lists; ``None`` for a legacy document without the field."""
    listed = source.get(ITEM_IDS_FIELD)
    return None if listed is None else [str(i) for i in listed]


def _age_s(source: dict[str, Any], field: str, now: datetime) -> float | None:
    """Seconds since the time stored in ``field``; ``None`` when it is absent or unreadable."""
    try:
        stamp = datetime.fromisoformat(str(source[field]))
    except (KeyError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return (now - stamp).total_seconds()


def repair_in_progress(source: dict[str, Any], *, now: datetime, repair_after_s: float) -> bool:
    """A repair of the image started less than ``repair_after_s`` ago and has not given its lease
    back."""
    lease = _age_s(source, REPAIR_LEASE_FIELD, now)
    return lease is not None and lease < repair_after_s


def repairable(source: dict[str, Any], *, now: datetime, repair_after_s: float) -> bool:
    """An incomplete photo no live writer can still be writing: no repair of it is in progress
    (a lease younger than ``repair_after_s``), and its writer marked the failure or its doc is
    older than ``repair_after_s``. An ``indexed_at`` that cannot be read counts as young (never
    repair on a guess)."""
    if repair_in_progress(source, now=now, repair_after_s=repair_after_s):
        return False
    if source.get(INCOMPLETE_MARK_FIELD):
        return True
    age = _age_s(source, INDEXED_AT_FIELD, now)
    return age is not None and age >= repair_after_s
