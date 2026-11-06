"""``db-migrate`` syncs the embedding model catalog after Alembic (2F-7): a project create fails
closed with ``DefaultEncoderMissingError`` when it never was, and nothing else would sync it."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from scripts.db import migrate
from src.db.repos import embedding_spaces as repo
from src.db.session import sync_deployment_tx
from src.services.embedding_spaces.catalog import load_catalog


pytestmark = pytest.mark.db


def _stored(pg) -> set[tuple[str, int]]:
    with (
        Session(pg.engine('app')) as session,
        sync_deployment_tx(session, 'test: read the catalog'),
    ):
        return {
            (e.model_key, e.version)
            for e in load_catalog()
            if repo.get_model_version(session, e.model_key, e.version) is not None
        }


def test_the_migrate_step_syncs_the_shipped_catalog_and_a_rerun_changes_nothing(pg_env) -> None:
    shipped = {(e.model_key, e.version) for e in load_catalog()}

    migrate.sync_embedding_catalog()
    first = _stored(pg_env)
    migrate.sync_embedding_catalog()

    assert first == shipped
    assert _stored(pg_env) == shipped
