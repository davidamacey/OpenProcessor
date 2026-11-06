"""Rows and fixtures shared by the embedding registry tests."""

from __future__ import annotations

import uuid
from typing import Any

import psycopg

from src.services.embedding_spaces.catalog import load_catalog


def shipped_engines() -> dict[str, dict[str, Any]]:
    """Ready-engine inventory that matches the shipped catalog (image contracts from the catalog)."""
    engines: dict[str, dict[str, Any]] = {}
    for entry in load_catalog():
        if entry.triton_image_model:
            engines[entry.triton_image_model] = entry.input_spec['image']
        if entry.triton_text_model:
            engines[entry.triton_text_model] = {}
    return engines


def make_project(conninfo: str, slug: str | None = None) -> tuple[uuid.UUID, str]:
    slug = slug or f'p{uuid.uuid4().hex[:12]}'
    project = uuid.uuid4()
    with psycopg.connect(conninfo, autocommit=True) as conn:
        conn.execute('INSERT INTO project_slugs (slug) VALUES (%s)', (slug,))
        conn.execute(
            'INSERT INTO projects (id, slug, display_name, status, index_prefix, storage_prefix, '
            "model_prefix, mlflow_experiment, created_by) VALUES (%s, %s, %s, 'active', %s, %s, %s, %s, 'test')",
            (project, slug, slug, f'{slug}__', f'projects/{slug}', f'{slug}__', slug),
        )
    return project, slug
