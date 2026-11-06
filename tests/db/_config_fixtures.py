"""Projects and sessions for the config-store database tests."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import psycopg


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase


def make_project(pg: PgDatabase, slug: str) -> uuid.UUID:
    """A project row with its settings and keymap rows, written as the superuser."""
    project = uuid.uuid4()
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        conn.execute('INSERT INTO project_slugs (slug) VALUES (%s)', (slug,))
        conn.execute(
            'INSERT INTO projects (id, slug, display_name, status, index_prefix, storage_prefix, '
            "model_prefix, mlflow_experiment, created_by) VALUES (%s, %s, %s, 'active', %s, %s, %s, %s, 'test')",
            (project, slug, slug, f'{slug}__', f'projects/{slug}', f'{slug}__', slug),
        )
        conn.execute(
            "INSERT INTO project_settings (project_id, updated_by) VALUES (%s, 'test')", (project,)
        )
        conn.execute(
            "INSERT INTO project_keymaps (project_id, updated_by) VALUES (%s, 'test')", (project,)
        )
    return project
