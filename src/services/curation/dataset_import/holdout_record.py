"""The durable record of an import's frozen test split: written once
when the import finishes as a ``holdout_freezes`` row that is not current, never replacing the
curated freeze."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from src.db.session import async_session_factory
from src.services.curation.holdout import forget_freeze, record_standing_freeze


if TYPE_CHECKING:
    from src.services.curation.dataset_import.context import ImportContext

_HOLDOUT_ACTIONS = frozenset({'created', 'updated', 'standalone', 'parent', 'proposal'})


async def persist_import_freeze(ctx: ImportContext, test_split: dict[str, Any]) -> uuid.UUID | None:
    """Record every item the import put on a ``test`` frame as holdout."""
    crop_ids = sorted(
        {
            item['crop_id']
            async for row in ctx.ledger.rows(statuses=('written',))
            if row.get('split') == 'test'
            for item in row.get('items') or []
            if item.get('action') in _HOLDOUT_ACTIONS
        }
    )
    if not crop_ids:
        return None
    async with async_session_factory()() as session:
        row = await record_standing_freeze(
            session,
            crop_ids=crop_ids,
            cohort_spec={
                'kind': 'import',
                'import_id': ctx.import_id,
                'source_sha': ctx.source_sha,
                **test_split,
            },
            per_class_counts={},
            source=f'import:{ctx.import_id}',
            frozen_by='import',
        )
    return row.id


async def mark_freeze_undone(freeze_record: str | None) -> None:
    """Drop the freeze row of an undone import: its items are no longer held out."""
    if freeze_record:
        async with async_session_factory()() as session:
            await forget_freeze(session, uuid.UUID(freeze_record))


__all__ = ['mark_freeze_undone', 'persist_import_freeze']
