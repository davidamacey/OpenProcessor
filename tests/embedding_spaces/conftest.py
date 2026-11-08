from __future__ import annotations

import pytest
from fixtures.job_fixtures import job_env, queue_hooks, wakes
from sqlalchemy.orm import Session

from src.services.embedding_spaces.catalog import sync_catalog


@pytest.fixture
def catalog(pg):
    """The shipped catalog synced into the module's database (idempotent)."""
    with Session(pg.engine('app')) as session:
        sync_catalog(session)


__all__ = ['catalog', 'job_env', 'queue_hooks', 'wakes']
