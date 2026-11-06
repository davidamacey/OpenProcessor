"""A created project has exactly its three default embedding spaces, written in the create
transaction (so a failed create leaves none)."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text

from src.db.session import async_session_factory, deployment_tx
from src.services.projects.bootstrap import bootstrap_default_project


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('pg_registry')]

_SPACES = text(
    'SELECT s.target, s.role, s.status, s.index_role, s.vector_field, s.state_field, '
    's.nested_path, v.model_key, v.version '
    'FROM embedding_spaces s JOIN embedding_model_versions v ON v.id = s.model_version_id '
    'WHERE s.project_id = :p ORDER BY s.target'
)


async def _spaces(project_id):
    async with async_session_factory()() as session, deployment_tx(session, 'test'):
        return [tuple(r) for r in (await session.execute(_SPACES, {'p': project_id})).all()]


def test_a_created_project_has_exactly_three_primary_ready_spaces() -> None:
    async def _run():
        record = await bootstrap_default_project()
        return await _spaces(record.id)

    rows = asyncio.run(_run())
    assert rows == [
        ('box', 'primary', 'ready', 'items', 'region_box_embeddings.embedding',
         None, 'region_box_embeddings', 'pe_core_l14_336', 1),
        ('frame', 'primary', 'ready', 'images', 'pe_embedding', None, None, 'pe_core_l14_336', 1),
        ('item', 'primary', 'ready', 'items', 'pe_embedding', 'embedding_state', None,
         'pe_core_l14_336', 1),
    ]  # fmt: skip


def test_a_failed_create_leaves_no_spaces_behind(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.embedding_spaces import spaces

    def boom(*_a, **_k):
        raise RuntimeError('space rows failed')

    monkeypatch.setattr(spaces, 'default_space_rows', boom)

    async def _run() -> int:
        with pytest.raises(RuntimeError, match='space rows failed'):
            await bootstrap_default_project()
        async with async_session_factory()() as session, deployment_tx(session, 'test'):
            projects = (await session.execute(text('SELECT count(*) FROM projects'))).scalar_one()
            spaces_n = (
                await session.execute(text('SELECT count(*) FROM embedding_spaces'))
            ).scalar_one()
        return projects + spaces_n

    assert asyncio.run(_run()) == 0
