"""The import ledger keeps per-image undo facts in a ``detail`` document that defaults to ``{}``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import psycopg
import pytest


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = pytest.mark.db


@pytest.fixture
def conn(pg: PgDatabase):
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as c:
        yield c


def test_the_column_is_a_required_jsonb_that_defaults_to_an_empty_object(conn) -> None:
    row = conn.execute(
        'SELECT data_type, is_nullable, column_default FROM information_schema.columns '
        "WHERE table_name = 'dataset_import_images' AND column_name = 'detail'"
    ).fetchone()
    assert row == ('jsonb', 'NO', "'{}'::jsonb")


def test_a_row_written_without_detail_reads_back_an_empty_object(conn) -> None:
    slug = 'detail-default'
    conn.execute('INSERT INTO project_slugs (slug) VALUES (%s)', (slug,))
    project = conn.execute(
        'INSERT INTO projects (slug, display_name, status, index_prefix, storage_prefix, '
        "model_prefix, mlflow_experiment, created_by) VALUES (%s, %s, 'active', %s, %s, %s, %s, 'test') "
        'RETURNING id',
        (slug, slug, f'{slug}__', f'projects/{slug}', f'{slug}__', slug),
    ).fetchone()[0]
    import_id = conn.execute(
        'INSERT INTO dataset_imports (project_id, source_key, format, mapping, options) '
        "VALUES (%s, 'src', 'coco', '{}', '{}') RETURNING id",
        (project,),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO dataset_import_images (import_id, image_key, status) VALUES (%s, 'a.jpg', 'pending')",
        (import_id,),
    )
    conn.execute(
        'INSERT INTO dataset_import_images (import_id, image_key, status, detail) '
        "VALUES (%s, 'b.jpg', 'written', %s::jsonb)",
        (import_id, '{"image_created": true, "boxes": [{"crop_id": "c1"}]}'),
    )
    rows = dict(
        conn.execute(
            'SELECT image_key, detail FROM dataset_import_images WHERE import_id = %s', (import_id,)
        ).fetchall()
    )
    assert rows['a.jpg'] == {}
    assert rows['b.jpg'] == {'image_created': True, 'boxes': [{'crop_id': 'c1'}]}
