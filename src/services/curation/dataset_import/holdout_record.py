"""The durable record of an import's frozen test split (W10.9): written once
when the import finishes, under the import freeze dir, never replacing the
curated ``current.json``."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import get_curation_config
from src.services.curation.holdout import (
    compute_holdout_sha,
    mark_freeze_record_undone,
    persist_freeze_record,
)


if TYPE_CHECKING:
    from src.services.curation.dataset_import.context import ImportContext
    from src.services.curation.dataset_import.store import ImportStore

_HOLDOUT_ACTIONS = frozenset({'created', 'updated', 'standalone', 'parent', 'proposal'})


def persist_import_freeze(ctx: ImportContext, store: ImportStore) -> Path | None:
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
    path = persist_freeze_record(
        crop_ids=crop_ids,
        sha=compute_holdout_sha(crop_ids),
        cohort_spec={
            'kind': 'import',
            'import_id': ctx.import_id,
            'source_sha': ctx.source_sha,
            **(state.get('test_split') or {}),
        },
        per_class_counts={},
        state_dir=Path(get_curation_config().project_state_dir) / 'test_holdout',
        kind='import',
    )
    store.job.update(freeze_record=str(path))
    return path


def mark_freeze_undone(store: ImportStore) -> None:
    recorded = store.job.read().get('freeze_record')
    if recorded:
        with contextlib.suppress(FileNotFoundError):
            mark_freeze_record_undone(Path(recorded))


__all__ = ['mark_freeze_undone', 'persist_import_freeze']
