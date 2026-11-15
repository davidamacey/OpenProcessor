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
the document is older than the repair age.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


ITEM_IDS_FIELD = 'item_ids'
INCOMPLETE_MARK_FIELD = 'ingest_incomplete'
"""Set by the writer that saw its own items step fail, on the images doc it created."""

INDEXED_AT_FIELD = 'indexed_at'


def listed_item_ids(source: dict[str, Any]) -> list[str] | None:
    """The items the images doc lists; ``None`` for a legacy document without the field."""
    listed = source.get(ITEM_IDS_FIELD)
    return None if listed is None else [str(i) for i in listed]


def repairable(source: dict[str, Any], *, now: datetime, repair_after_s: float) -> bool:
    """An incomplete photo no live writer can still be writing: marked by its writer, or older
    than ``repair_after_s``. An ``indexed_at`` that cannot be read counts as young (never repair
    on a guess)."""
    if source.get(INCOMPLETE_MARK_FIELD):
        return True
    try:
        indexed = datetime.fromisoformat(str(source[INDEXED_AT_FIELD]))
    except (KeyError, ValueError):
        return False
    if indexed.tzinfo is None:
        indexed = indexed.replace(tzinfo=UTC)
    return (now - indexed).total_seconds() >= repair_after_s
