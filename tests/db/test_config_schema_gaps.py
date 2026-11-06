"""Config-store schema decisions taken when the config services moved to PostgreSQL.

- Turning an axis off records an activation event whose target is NULL (no sentinel id), and
  the target id and revision are both NULL or both set.
- A clone records the source document and the source revision.
- Each revision keeps its own description (the document's description is the current one).
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa


pytestmark = pytest.mark.db


def _nullable(conn: sa.Connection, table: str) -> dict[str, bool]:
    rows = conn.execute(
        sa.text(
            'SELECT column_name, is_nullable FROM information_schema.columns WHERE table_name = :t'
        ),
        {'t': table},
    )
    return {name: nullable == 'YES' for name, nullable in rows}


def _check_source(conn: sa.Connection, table: str, name: str) -> str | None:
    return conn.execute(
        sa.text(
            'SELECT pg_get_constraintdef(oid) FROM pg_constraint '
            'WHERE conrelid = CAST(:t AS regclass) AND conname = :n'
        ),
        {'t': table, 'n': name},
    ).scalar_one_or_none()


def test_an_off_event_has_a_null_target_and_the_pair_moves_together(pg) -> None:
    with pg.engine('owner').connect() as conn:
        cols = _nullable(conn, 'config_activation_events')
        pair = _check_source(
            conn, 'config_activation_events', 'ck_config_activation_events_to_pair'
        )
    assert cols['to_config_id'] is True
    assert cols['to_revision'] is True
    assert pair is not None
    assert 'to_config_id IS NULL' in pair
    assert 'to_revision IS NULL' in pair


def test_a_clone_records_its_source_revision(pg) -> None:
    with pg.engine('owner').connect() as conn:
        cols = _nullable(conn, 'config_documents')
    assert cols.get('cloned_from_revision') is True


def test_each_revision_keeps_its_own_description(pg) -> None:
    with pg.engine('owner').connect() as conn:
        cols = _nullable(conn, 'config_revisions')
        default = conn.execute(
            sa.text(
                'SELECT column_default FROM information_schema.columns '
                "WHERE table_name = 'config_revisions' AND column_name = 'description'"
            )
        ).scalar_one_or_none()
    assert cols.get('description') is False
    assert default is not None
    assert "''" in default
