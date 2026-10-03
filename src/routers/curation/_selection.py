"""Bulk item writes take explicit ``crop_ids`` or a selection (the shared item
filter, optionally capped), resolved once so a dry run counts exactly what the
write then changes."""

from __future__ import annotations

from typing import Any, Self

from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator

from src.config import get_curation_config
from src.services.curation.item_selection import (
    MAX_SELECTION_IDS,
    ItemSelection,
    SelectionError,
    resolve_selection,
)


# A filter selection above this is refused, not truncated: add a limit or a tighter filter.
MAX_BULK_SELECTION = 20_000


class SelectionTargets(BaseModel):
    """Exactly one of ``crop_ids`` or ``selection``. ``dry_run`` reports how many
    items the request would change and writes nothing."""

    crop_ids: list[str] | None = Field(default=None, max_length=MAX_SELECTION_IDS)
    selection: ItemSelection | None = None
    dry_run: bool = False

    @model_validator(mode='after')
    def _one_target(self) -> Self:
        if (self.crop_ids is None) == (self.selection is None):
            msg = 'exactly one of crop_ids or selection is required'
            raise ValueError(msg)
        return self


async def selected_crop_ids(opensearch: Any, targets: SelectionTargets) -> list[str]:
    """The ids the request names (``422`` when a selection is malformed or
    larger than :data:`MAX_BULK_SELECTION`)."""
    if targets.crop_ids is not None:
        return list(dict.fromkeys(targets.crop_ids))
    assert targets.selection is not None
    try:
        ids = await resolve_selection(
            opensearch, targets.selection, index=get_curation_config().items_index
        )
    except SelectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if len(ids) > MAX_BULK_SELECTION:
        raise HTTPException(
            status_code=422,
            detail=f'the selection names {len(ids)} items; the limit is {MAX_BULK_SELECTION} (set limit)',
        )
    return ids
