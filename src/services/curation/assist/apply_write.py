"""The one write of a class onto items from an assist decision.

An explicit apply and the auto-label loop's propagation both write through
:func:`write_class_to_ids`, so for the same label they produce the same
document. Every item is re-checked under the OCC read-merge-write: a class a
human or an import owns, a validated class and a holdout item are never
overwritten, and with ``tokens`` an item whose class state moved since it was
read is left alone too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.services.curation.class_label import ItemLabel, class_label_update
from src.services.curation.class_write_guard import class_state_token, class_write_locked
from src.services.curation.crop_bulk import occ_bulk_human_relabel


if TYPE_CHECKING:
    from collections.abc import Mapping

    from opensearchpy import AsyncOpenSearch


@dataclass
class ClassWrite:
    updated_ids: list[str]
    conflicts: list[dict[str, Any]]
    """Items the OCC write could not land after its retries."""
    skipped_locked: list[str]
    """Items locked or validated by the time of the write."""
    skipped_changed: list[str]
    """Items whose class state differs from the token the caller read."""


async def write_class_to_ids(
    opensearch: AsyncOpenSearch,
    ids: list[str],
    label: ItemLabel,
    *,
    tokens: Mapping[str, tuple[Any, ...]] | None = None,
) -> ClassWrite:
    """Write ``label`` to ``ids``. ``tokens`` maps an id to the
    :func:`~src.services.curation.class_write_guard.class_state_token` its decision
    was made on; with it, an id without a matching token is skipped (fail closed)."""
    locked: set[str] = set()
    changed: set[str] = set()

    def merge(current: dict[str, Any]) -> dict[str, Any]:
        crop_id = str(current.get('crop_id', ''))
        if class_write_locked(current):
            locked.add(crop_id)
            return {}
        if tokens is not None and tokens.get(crop_id) != class_state_token(current):
            changed.add(crop_id)
            return {}
        return class_label_update(current, label)

    written = await occ_bulk_human_relabel(opensearch, ids, merge, writer_id=label.writer)
    skipped = locked | changed
    return ClassWrite(
        updated_ids=[i for i in written['updated_ids'] if i not in skipped],
        conflicts=list(written['conflicts']),
        skipped_locked=[i for i in ids if i in locked],
        skipped_changed=[i for i in ids if i in changed],
    )


__all__ = ['ClassWrite', 'write_class_to_ids']
