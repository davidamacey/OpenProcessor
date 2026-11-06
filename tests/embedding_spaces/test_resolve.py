"""The reader lookup: default field names, cache refresh on the config epoch, fail closed."""

from __future__ import annotations

import pytest

from src.config.project_context import ProjectNotBound, bind_project
from src.db.repos import embedding_spaces as repo
from src.db.session import async_session_factory, project_tx
from src.domain.embedding_spaces import EmbeddingSpaceError, Target
from src.services.embedding_spaces import resolve, spaces
from src.services.embedding_spaces.flip import promote
from tests.embedding_spaces._seed import make_project, shipped_engines


pytestmark = [pytest.mark.db, pytest.mark.asyncio, pytest.mark.usefixtures('bound_async_engine')]


class _Record:
    def __init__(self, slug: str) -> None:
        self.slug = slug


async def _noop(_role: str, _body: dict) -> None:
    return None


@pytest.fixture(autouse=True)
def _fresh_cache():
    resolve.clear_cache()
    yield
    resolve.clear_cache()


async def _seed_defaults(project) -> None:
    async with async_session_factory()() as session, project_tx(session, project):
        version = await session.run_sync(lambda s: repo.get_model_version(s, 'pe_core_l14_336', 1))
        await repo.add_spaces(
            session, spaces.default_space_rows(project, version, created_by='test')
        )


async def test_a_fresh_project_resolves_the_default_field_names(pg_env, catalog) -> None:
    project, slug = make_project(pg_env.conninfo('super'))
    await _seed_defaults(project)
    with bind_project(_Record(slug)):  # type: ignore[arg-type]
        item = await resolve.embedding_space(Target.ITEM)
        box = await resolve.embedding_space(Target.BOX)
        frame = await resolve.embedding_space(Target.FRAME)
    assert (item.vector_field, item.state_field, item.space_key) == (
        'pe_embedding',
        'embedding_state',
        'pe_core_l14_336_v1',
    )
    assert (box.nested_path, box.vector_field) == (
        'region_box_embeddings',
        'region_box_embeddings.embedding',
    )
    assert (frame.index_role, frame.vector_field, frame.state_field) == (
        'images',
        'pe_embedding',
        None,
    )
    assert item.dim == 1024
    assert item.triton_text_model == 'pe_text_encoder'


async def test_a_flip_is_seen_after_one_epoch_poll(pg_env, catalog) -> None:
    project, slug = make_project(pg_env.conninfo('super'))
    await _seed_defaults(project)
    with bind_project(_Record(slug)):  # type: ignore[arg-type]
        assert (await resolve.embedding_space(Target.ITEM)).space_key == 'pe_core_l14_336_v1'
        async with async_session_factory()() as session:
            row = await spaces.register_space(
                session,
                project,
                model_key='mobileclip2_s2',
                version=1,
                target=Target.ITEM,
                created_by='test',
                ready_engines=shipped_engines(),
            )
            await spaces.begin_build(
                session, project, target=Target.ITEM, key=row.space_key, put_mapping=_noop
            )
            await spaces.mark_ready(session, project, target=Target.ITEM, key=row.space_key)
            flipped = await promote(
                session, project, target=Target.ITEM, key=row.space_key, force=True
            )
        # the poll reports the old epoch: the cache is still valid
        resolve.note_config_epoch(slug, flipped.config_epoch - 1)
        assert (await resolve.embedding_space(Target.ITEM)).space_key == 'pe_core_l14_336_v1'
        resolve.note_config_epoch(slug, flipped.config_epoch)
        now = await resolve.embedding_space(Target.ITEM)
        named = await resolve.embedding_space(Target.ITEM, 'pe_core_l14_336_v1')
    assert now.space_key == 'mobileclip2_s2_v1'
    assert named.space_key == 'pe_core_l14_336_v1'


async def test_without_a_poll_the_epoch_is_read_each_call(pg_env, catalog) -> None:
    project, slug = make_project(pg_env.conninfo('super'))
    await _seed_defaults(project)
    with bind_project(_Record(slug)):  # type: ignore[arg-type]
        await resolve.embedding_space(Target.ITEM)
        async with async_session_factory()() as session:
            row = await spaces.register_space(
                session,
                project,
                model_key='mobileclip2_s2',
                version=1,
                target=Target.ITEM,
                created_by='test',
                ready_engines=shipped_engines(),
            )
            await spaces.begin_build(
                session, project, target=Target.ITEM, key=row.space_key, put_mapping=_noop
            )
            await spaces.mark_ready(session, project, target=Target.ITEM, key=row.space_key)
            await promote(session, project, target=Target.ITEM, key=row.space_key, force=True)
        assert (await resolve.embedding_space(Target.ITEM)).space_key == 'mobileclip2_s2_v1'


@pytest.mark.unbound
async def test_an_unbound_call_raises(pg_env, catalog) -> None:
    with pytest.raises(ProjectNotBound):
        await resolve.embedding_space(Target.ITEM)


async def test_an_unknown_space_is_a_404(pg_env, catalog) -> None:
    project, slug = make_project(pg_env.conninfo('super'))
    await _seed_defaults(project)
    with bind_project(_Record(slug)):  # type: ignore[arg-type]
        with pytest.raises(EmbeddingSpaceError) as err:
            await resolve.embedding_space(Target.ITEM, 'nope_v9')
        with pytest.raises(EmbeddingSpaceError):
            await resolve.embedding_space(Target.ITEM_TEXT)
    assert err.value.status == 404
