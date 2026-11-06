"""The durable record of an import's frozen test split: written once
when the import finishes as a ``holdout_freezes`` row that is not current, never replacing the
curated freeze."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from src.db.session import async_session_factory
from src.services.curation.holdout import forget_freeze, record_standing_freeze


if TYPE_CHECKING:
    from src.services.curation.dataset_import.context import ImportContext
    from src.services.curation.dataset_import.store import ImportStore

_HOLDOUT_ACTIONS = frozenset({'created', 'updated', 'standalone', 'parent', 'proposal'})


async def persist_import_freeze(ctx: ImportContext, store: ImportStore) -> uuid.UUID | None:
    """Record every item the import put on a ``test`` frame as holdout."""
    crop_ids = sorted(
        {
            item['crop_id']
            for row in store.ledger_rows()
            if row.get('split') == 'test' and row.get('status') == 'ok'
            for item in row.get('items') or []
            if item.get('action') in _HOLDOUT_ACTIONS
        }
    )
    if not crop_ids:
        return None
    state = store.job.read()
    async with async_session_factory()() as session:
        row = await record_standing_freeze(
            session,
            crop_ids=crop_ids,
            cohort_spec={
                'kind': 'import',
                'import_id': ctx.import_id,
                'source_sha': ctx.source_sha,
                **(state.get('test_split') or {}),
            },
            per_class_counts={},
            source=f'import:{ctx.import_id}',
            frozen_by='import',
        )
    store.job.update(freeze_record=str(row.id))
    return row.id


async def mark_freeze_undone(store: ImportStore) -> None:
    """Drop the freeze row of an undone import: its items are no longer held out."""
    recorded = store.job.read().get('freeze_record')
    if recorded:
        async with async_session_factory()() as session:
            await forget_freeze(session, uuid.UUID(recorded))


__all__ = ['mark_freeze_undone', 'persist_import_freeze']
