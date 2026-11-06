"""A project row for the registry tests, written as the superuser (past row-level security)."""

from __future__ import annotations

import uuid
from typing import Any


def make_project(conn: Any, slug: str) -> uuid.UUID:
    project = uuid.uuid4()
    conn.execute('INSERT INTO project_slugs (slug) VALUES (%s)', (slug,))
    conn.execute(
        'INSERT INTO projects (id, slug, display_name, status, index_prefix, storage_prefix, '
        "model_prefix, mlflow_experiment, created_by) VALUES (%s, %s, %s, 'active', %s, %s, %s, %s, 'test')",
        (project, slug, slug, f'{slug}__', f'projects/{slug}', f'{slug}__', slug),
    )
    return project
