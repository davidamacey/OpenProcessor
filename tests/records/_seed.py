"""A project row written as the superuser, for the records tests."""

from __future__ import annotations

import uuid

import psycopg


def make_project(pg, slug: str) -> uuid.UUID:
    project = uuid.uuid4()
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        conn.execute('INSERT INTO project_slugs (slug) VALUES (%s)', (slug,))
        conn.execute(
            'INSERT INTO projects (id, slug, display_name, status, index_prefix, storage_prefix, '
            "model_prefix, mlflow_experiment, created_by) VALUES (%s, %s, %s, 'active', %s, %s, %s, %s, 'test')",
            (project, slug, slug, f'{slug}__', f'projects/{slug}', f'{slug}__', slug),
        )
    return project
