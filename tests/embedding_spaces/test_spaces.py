"""Registering, building, flipping and retiring spaces against a real PostgreSQL."""

from __future__ import annotations

from dataclasses import replace

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.db.repos import embedding_spaces as repo
from src.db.session import async_session_factory, project_tx
from src.domain.embedding_spaces import EmbeddingSpaceError, KnnMethodSpec, Role, Status, Target
from src.services.embedding_spaces import spaces
from src.services.embedding_spaces.flip import promote
from tests.embedding_spaces._seed import make_project, shipped_engines


pytestmark = [pytest.mark.db, pytest.mark.asyncio, pytest.mark.usefixtures('bound_async_engine')]

ENGINES = shipped_engines()


async def _noop_mapping(_role: str, _body: dict) -> None:
    return None


async def _register(session, project, key='mobileclip2_s2', target=Target.ITEM, **kw):
    return await spaces.register_space(
        session,
        project,
        model_key=key,
        version=1,
        target=target,
        created_by='test',
        ready_engines=ENGINES,
        **kw,
    )


async def _ready(session, project, key='mobileclip2_s2', target=Target.ITEM):
    row = await _register(session, project, key, target)
    await spaces.begin_build(
        session, project, target=target, key=row.space_key, put_mapping=_noop_mapping
    )
    return await spaces.mark_ready(session, project, target=target, key=row.space_key)


async def _default_primary(session, project, target=Target.ITEM):
    """The project create's default spaces: ``pe_core_l14_336`` v1, primary and ready."""
    async with project_tx(session, project):
        version = await session.run_sync(lambda s: repo.get_model_version(s, 'pe_core_l14_336', 1))
        rows = spaces.default_space_rows(project, version, created_by='test')
        await repo.add_spaces(session, rows)


async def test_register_and_read_back(pg_env, catalog) -> None:
    project, _ = make_project(pg_env.conninfo('super'))
    async with async_session_factory()() as session:
        row = await _register(session, project)
        assert (row.space_key, row.status, row.role) == (
            'mobileclip2_s2_v1',
            Status.PENDING,
            Role.CANDIDATE,
        )
        assert (row.vector_field, row.state_field) == (
            'vec_mobileclip2_s2_v1',
            'vec_mobileclip2_s2_v1_state',
        )
        listed = await spaces.list_project_spaces(session, project)
    assert [r.space_key for r in listed] == ['mobileclip2_s2_v1']


async def _add_second_primary(session, project) -> None:
    async with project_tx(session, project):
        version = await session.run_sync(lambda s: repo.get_model_version(s, 'mobileclip2_s2', 1))
        row = replace(
            spaces.default_space_rows(project, version, created_by='test')[0],
            space_key='alt_v1',
            vector_field='vec_alt_v1',
            state_field='vec_alt_v1_state',
        )
        await repo.add_space(session, row)


async def test_a_second_primary_for_one_target_is_a_unique_violation(pg_env, catalog) -> None:
    project, _ = make_project(pg_env.conninfo('super'))
    async with async_session_factory()() as session:
        await _default_primary(session, project)
        with pytest.raises(IntegrityError, match='uq_embedding_spaces_one_primary'):
            await _add_second_primary(session, project)


async def test_promote_of_a_building_space_answers_409(pg_env, catalog) -> None:
    project, _ = make_project(pg_env.conninfo('super'))
    async with async_session_factory()() as session:
        await _default_primary(session, project)
        row = await _register(session, project)
        await spaces.begin_build(
            session, project, target=Target.ITEM, key=row.space_key, put_mapping=_noop_mapping
        )
        with pytest.raises(EmbeddingSpaceError) as err:
            await promote(session, project, target=Target.ITEM, key=row.space_key)
    assert (err.value.code, err.value.status) == ('embedding_space_not_ready', 409)


async def test_promote_needs_an_evaluation_unless_forced_and_bumps_the_epoch(
    pg_env, catalog
) -> None:
    project, _ = make_project(pg_env.conninfo('super'))
    async with async_session_factory()() as session:
        await _default_primary(session, project)
        row = await _ready(session, project)
        async with project_tx(session, project):
            before = await session.scalar(
                text('SELECT config_epoch FROM projects WHERE id = :p'), {'p': project}
            )
        with pytest.raises(EmbeddingSpaceError) as err:
            await promote(session, project, target=Target.ITEM, key=row.space_key)
        assert err.value.code == 'embedding_evaluation_missing'
        result = await promote(session, project, target=Target.ITEM, key=row.space_key, force=True)
        assert result.primary.role is Role.PRIMARY
        assert result.previous is not None
        assert (result.previous.space_key, result.previous.role) == (
            'pe_core_l14_336_v1',
            Role.OPTIONAL,
        )
        assert result.config_epoch == before + 1
        async with project_tx(session, project):
            primary = await repo.get_primary_space(session, project, Target.ITEM)
        assert primary is not None
        assert primary.space_key == 'mobileclip2_s2_v1'


async def test_an_evaluation_run_lets_the_flip_through_without_force(pg_env, catalog) -> None:
    project, _ = make_project(pg_env.conninfo('super'))
    async with async_session_factory()() as session:
        await _default_primary(session, project)
        row = await _ready(session, project)
        async with project_tx(session, project):
            old = await repo.get_primary_space(session, project, Target.ITEM)
            await session.execute(
                text(
                    'INSERT INTO embedding_evaluation_runs (project_id, baseline_space_id, '
                    'candidate_space_id, protocol, sample_spec, metrics, verdict, created_by) '
                    "VALUES (:p, :b, :c, 'query_set_v1', '{}', '{}', 'better', 'test')"
                ),
                {'p': project, 'b': old.id, 'c': row.id},
            )
        result = await promote(session, project, target=Target.ITEM, key=row.space_key)
    assert result.primary.space_key == row.space_key


async def test_at_most_three_live_spaces_per_target(pg_env, catalog) -> None:
    project, _ = make_project(pg_env.conninfo('super'))
    async with async_session_factory()() as session:
        await _register(session, project, 'mobileclip2_s2')
        await _register(session, project, 'pe_core_l14_336')
        # the catalog has two models, so the third live item space is written directly
        async with project_tx(session, project):
            version = await session.run_sync(
                lambda s: repo.get_model_version(s, 'mobileclip2_s2', 1)
            )
            await repo.add_space(
                session,
                repo.NewSpace(
                    project_id=project,
                    space_key='extra_v1',
                    model_version_id=version.id,
                    target=Target.ITEM,
                    dim=512,
                    role=Role.OPTIONAL,
                    status=Status.PENDING,
                    index_role='items',
                    vector_field='vec_extra_v1',
                    state_field=None,
                    nested_path=None,
                    knn_method=KnnMethodSpec(),
                    created_by='test',
                ),
            )
        with pytest.raises(EmbeddingSpaceError) as err:
            await _register(session, project, 'mobileclip2_s2', dim=None, role=Role.OPTIONAL)
    assert err.value.code == 'embedding_space_limit'


async def test_a_model_without_a_ready_engine_cannot_back_a_space(pg_env, catalog) -> None:
    project, _ = make_project(pg_env.conninfo('super'))
    async with async_session_factory()() as session:
        with pytest.raises(EmbeddingSpaceError) as err:
            await spaces.register_space(
                session,
                project,
                model_key='mobileclip2_s2',
                version=1,
                target=Target.ITEM,
                created_by='test',
                ready_engines={},
            )
    assert err.value.code == 'embedding_model_unavailable'


async def test_the_primary_cannot_be_retired_and_an_optional_one_can(pg_env, catalog) -> None:
    project, _ = make_project(pg_env.conninfo('super'))
    async with async_session_factory()() as session:
        await _default_primary(session, project)
        with pytest.raises(EmbeddingSpaceError) as err:
            await spaces.retire_space(
                session, project, target=Target.ITEM, key='pe_core_l14_336_v1'
            )
        assert err.value.code == 'embedding_space_primary'
        row = await _ready(session, project)
        moved = await spaces.retire_space(session, project, target=Target.ITEM, key=row.space_key)
        assert moved.status is Status.RETIRING
        done = await spaces.finish_retire(session, project, target=Target.ITEM, key=row.space_key)
        assert done.status is Status.RETIRED
        assert done.retired_at is not None


async def test_begin_build_puts_the_new_fields_and_a_refusal_keeps_the_space_pending(
    pg_env, catalog
) -> None:
    project, _ = make_project(pg_env.conninfo('super'))
    seen: list[tuple[str, dict]] = []

    async def record(role: str, body: dict) -> None:
        seen.append((role, body))

    async def refuse(_role: str, _body: dict) -> None:
        raise RuntimeError('mapping refused')

    async with async_session_factory()() as session:
        row = await _register(session, project, target=Target.BOX)
        with pytest.raises(RuntimeError, match='refused'):
            await spaces.begin_build(
                session, project, target=Target.BOX, key=row.space_key, put_mapping=refuse
            )
        still = await spaces.list_project_spaces(session, project, Target.BOX)
        assert still[0].status is Status.PENDING
        await spaces.begin_build(
            session, project, target=Target.BOX, key=row.space_key, put_mapping=record
        )
    role, body = seen[0]
    assert role == 'items'
    assert list(body) == ['boxvec_mobileclip2_s2_v1']
    nested = body['boxvec_mobileclip2_s2_v1']
    assert nested['type'] == 'nested'
    assert nested['properties']['embedding']['dimension'] == 512
