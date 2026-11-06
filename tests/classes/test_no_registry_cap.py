"""There is no 10,000-class cap and the retired registry error codes are gone (2B-5, D-57)."""

from __future__ import annotations

import typing
import uuid

import psycopg
import pytest
from _projects import make_project

from src.schemas.curation.config_common import ErrorCode
from src.services.classes import registry as registry_module
from src.services.classes.registry import ClassRegistry


def test_the_retired_registry_error_codes_are_gone() -> None:
    codes = set(typing.get_args(ErrorCode))
    assert not codes & {'registry_not_migrated', 'registry_full', 'registry_conflict'}
    assert not hasattr(registry_module, 'MAX_REGISTRY_CLASSES')


@pytest.mark.db
@pytest.mark.asyncio
async def test_ten_thousand_and_one_classes_are_accepted(
    pg, bound_async_engine, monkeypatch
) -> None:
    slug = f'p{uuid.uuid4().hex[:8]}'
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        make_project(conn, slug)
        # 10,000 classes straight into the table, then the service adds the 10,001st.
        pid = conn.execute('SELECT id FROM projects WHERE slug = %s', (slug,)).fetchone()[0]
        conn.execute(
            'INSERT INTO class_registries (project_id, next_class_id) VALUES (%s, 10000)', (pid,)
        )
        conn.execute(
            'INSERT INTO project_classes (project_id, class_id, name, name_key) '
            "SELECT %s, g, 'class ' || g, 'class_' || g FROM generate_series(0, 9999) g",
            (pid,),
        )
    monkeypatch.setattr(registry_module, '_project_ids', {})
    reg = ClassRegistry(slug)

    new_id = await reg.add_class('one more')

    assert new_id == 10_000
    assert len((await reg.snapshot()).classes) == 10_001
