"""Test-only: what the dataset-import tests need from the near-duplicate lane (ND-A).

The ledger's ``detail`` column and ``enqueue_pending`` come with ND-A's revision and module,
which this lane's base does not carry.

``dataset_import_images.detail`` is added by the near-duplicate lane's schema revision (0003),
which these tests' base does not carry. Until that merge, every test that writes the ledger adds
the column to its test database here. This depends on the ND-A merge: delete this module (and its
callers) when the revision lands.
"""

from __future__ import annotations

import sys
import types
from typing import Any

from fixtures.job_rows import sql


def ensure_detail_column(pg: Any) -> None:
    sql(
        pg,
        'ALTER TABLE dataset_import_images ADD COLUMN IF NOT EXISTS detail jsonb NOT NULL '
        "DEFAULT '{}'::jsonb",
    )


def stand_in_enqueue_pending(monkeypatch: Any) -> list[tuple[str, list[str], str]]:
    """ND-A's ``enqueue_pending(session, project_id, level, member_ids, reason)``, recorded.

    Stood in for in ``sys.modules`` for the length of one test, never in ``src/``."""
    calls: list[tuple[str, list[str], str]] = []

    async def enqueue_pending(_session: Any, _project: Any, level: str, ids: Any, reason: str):
        calls.append((level, list(ids), reason))

    package = types.ModuleType('src.services.near_duplicates')
    package.__path__ = []  # type: ignore[attr-defined]
    module = types.ModuleType('src.services.near_duplicates.pending')
    module.enqueue_pending = enqueue_pending  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'src.services.near_duplicates', package)
    monkeypatch.setitem(sys.modules, 'src.services.near_duplicates.pending', module)
    return calls
