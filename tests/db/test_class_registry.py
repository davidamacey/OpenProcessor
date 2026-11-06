"""The class registry repository against a real PostgreSQL (02a section 3, 02c section 7)."""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid

import psycopg
import pytest
from _research_seed import make_project
from sqlalchemy import text

from src.db.repos import classes as repo
from src.db.repos.classes import (
    ClassLoad,
    ClassNameInvalidError,
    ClassNameTakenError,
    ClassRegistryError,
    HotkeyTakenError,
)
from src.db.session import async_session_factory, project_tx


pytestmark = [pytest.mark.db, pytest.mark.usefixtures('bound_async_engine')]


@pytest.fixture
def project(pg, pg_env) -> uuid.UUID:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        return make_project(conn, f'p{uuid.uuid4().hex[:8]}')


async def _in_tx(project: uuid.UUID, fn):
    async with async_session_factory()() as session, project_tx(session, project):
        return await fn(session)


@pytest.mark.asyncio
async def test_sequential_adds_give_ids_0_1_2(project) -> None:
    async def run(s):
        return [(await repo.add_class(s, project, n)).class_id for n in ('a', 'b', 'c')]

    assert await _in_tx(project, run) == [0, 1, 2]
    rows = await _in_tx(project, lambda s: repo.read_registry(s, project))
    assert rows.next_class_id == 3
    assert rows.revision == 3


@pytest.mark.asyncio
async def test_ten_concurrent_ensure_class_calls_for_fifty_names_make_fifty_classes(
    project,
) -> None:
    names = [f'class {i}' for i in range(50)]

    async def worker():
        async def run(s):
            return [(await repo.ensure_class(s, project, n))[0].class_id for n in names]

        return await _in_tx(project, run)

    results = await asyncio.gather(*(worker() for _ in range(10)))
    rows = await _in_tx(project, lambda s: repo.read_registry(s, project))
    assert len(rows.classes) == 50
    assert sorted(c.class_id for c in rows.classes) == list(range(50))
    assert rows.next_class_id == 50
    assert all(r == results[0] for r in results)


@pytest.mark.asyncio
async def test_car_variants_are_one_class_and_traffic_light_equals_traffic_underscore_light(
    project,
) -> None:
    async def run(s):
        ids = {(await repo.ensure_class(s, project, n))[0].class_id for n in ('Car', 'car', 'car ')}
        a = (await repo.ensure_class(s, project, 'traffic light'))[0].class_id
        b = (await repo.ensure_class(s, project, 'traffic_light'))[0].class_id
        return ids, a, b

    ids, a, b = await _in_tx(project, run)
    assert len(ids) == 1
    assert a == b


@pytest.mark.asyncio
async def test_add_class_refuses_an_active_duplicate_by_key(project) -> None:
    async def run(s):
        await repo.add_class(s, project, 'Car')
        with pytest.raises(ClassNameTakenError):
            await repo.add_class(s, project, 'car ')

    await _in_tx(project, run)


@pytest.mark.asyncio
async def test_a_deprecated_name_and_a_new_active_one_coexist_and_ensure_picks_the_active(
    project,
) -> None:
    async def run(s):
        old = (await repo.add_class(s, project, 'bus')).class_id
        await repo.set_deprecated(s, project, old, True)
        new, created = await repo.ensure_class(s, project, 'Bus')
        return old, new, created

    old, new, created = await _in_tx(project, run)
    assert created
    assert new.class_id != old
    assert not new.deprecated


@pytest.mark.asyncio
async def test_restoring_a_deprecated_name_while_an_active_one_exists_is_class_name_taken(
    project,
) -> None:
    async def run(s):
        old = (await repo.add_class(s, project, 'bus')).class_id
        await repo.set_deprecated(s, project, old, True)
        await repo.ensure_class(s, project, 'bus')
        with pytest.raises(ClassNameTakenError):
            await repo.set_deprecated(s, project, old, False)
        # the savepoint kept the transaction usable
        return await repo.read_registry(s, project)

    rows = await _in_tx(project, run)
    assert [c.deprecated for c in rows.classes] == [True, False]


@pytest.mark.asyncio
async def test_a_name_with_no_slug_safe_character_is_refused(project) -> None:
    async def run(s):
        for fn in (repo.ensure_class, repo.add_class):
            with pytest.raises(ClassNameInvalidError):
                await fn(s, project, ' !! ')

    await _in_tx(project, run)


@pytest.mark.asyncio
async def test_a_duplicate_hotkey_is_a_conflict(project) -> None:
    async def run(s):
        await repo.add_class(s, project, 'a', hotkey_letter='q')
        with pytest.raises(HotkeyTakenError):
            await repo.add_class(s, project, 'b', hotkey_letter='q')
        b = (await repo.add_class(s, project, 'b')).class_id
        with pytest.raises(HotkeyTakenError):
            await repo.update_class(s, project, b, hotkey_letter='q', set_hotkey=True)

    await _in_tx(project, run)


@pytest.mark.asyncio
async def test_set_hotkeys_swaps_two_letters(project) -> None:
    async def run(s):
        a = (await repo.add_class(s, project, 'a', hotkey_letter='q')).class_id
        b = (await repo.add_class(s, project, 'b', hotkey_letter='w')).class_id
        await repo.set_hotkeys(s, project, {a: 'w', b: 'q'})
        return a, b, await repo.read_registry(s, project)

    a, b, rows = await _in_tx(project, run)
    by_id = {c.class_id: c.hotkey_letter for c in rows.classes}
    assert by_id == {a: 'w', b: 'q'}


@pytest.mark.asyncio
async def test_merged_into_must_be_deprecated_and_merge_deprecates_the_source(project) -> None:
    async def run(s):
        a = (await repo.add_class(s, project, 'a')).class_id
        b = (await repo.add_class(s, project, 'b')).class_id
        await repo.merge_class(s, project, a, b)
        with pytest.raises(ClassRegistryError, match='deprecated'):
            await repo.merge_class(s, project, b, a)
        return a, await repo.get_class(s, project, a)

    _a, source = await _in_tx(project, run)
    assert source.deprecated
    assert source.merged_into is not None


@pytest.mark.asyncio
async def test_the_database_refuses_a_merge_target_on_a_live_class(pg, project) -> None:
    async def run(s):
        a = (await repo.add_class(s, project, 'a')).class_id
        b = (await repo.add_class(s, project, 'b')).class_id
        return a, b

    a, b = await _in_tx(project, run)
    with (
        psycopg.connect(pg.conninfo('super'), autocommit=True) as conn,
        pytest.raises(psycopg.errors.CheckViolation),
    ):
        conn.execute(
            'UPDATE project_classes SET merged_into = %s WHERE project_id = %s AND class_id = %s',
            (b, project, a),
        )


@pytest.mark.asyncio
async def test_every_write_bumps_the_revision(project) -> None:
    async def run(s):
        seen = []
        a = (await repo.add_class(s, project, 'a')).class_id
        seen.append(await repo.current_revision(s, project))
        await repo.update_class(s, project, a, group='g')
        seen.append(await repo.current_revision(s, project))
        await repo.set_deprecated(s, project, a, True)
        seen.append(await repo.current_revision(s, project))
        return seen

    assert await _in_tx(project, run) == [1, 2, 3]


@pytest.mark.asyncio
async def test_a_rename_and_a_merge_write_their_outbox_rows_in_the_same_transaction(
    project,
) -> None:
    async def run(s):
        a = (await repo.add_class(s, project, 'a')).class_id
        b = (await repo.add_class(s, project, 'b')).class_id
        await repo.update_class(s, project, a, name='alpha')
        await repo.merge_class(s, project, a, b)
        rows = (
            await s.execute(
                text('SELECT kind, payload FROM search_projection_tasks ORDER BY created_at, kind')
            )
        ).all()
        return a, b, rows

    a, b, rows = await _in_tx(project, run)
    by_kind = dict(rows)
    assert by_kind['class_rename'] == {'class_id': a, 'new_name': 'alpha'}
    assert by_kind['class_merge'] == {'source_ids': [a], 'target_id': b}


@pytest.mark.asyncio
async def test_a_rolled_back_rename_leaves_no_outbox_row(project) -> None:
    async def setup(s):
        return (await repo.add_class(s, project, 'a')).class_id

    a = await _in_tx(project, setup)

    async def rename_then_fail(s):
        await repo.update_class(s, project, a, name='alpha')
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await _in_tx(project, rename_then_fail)

    async def check(s):
        n = (await s.execute(text('SELECT count(*) FROM search_projection_tasks'))).scalar_one()
        return n, (await repo.get_class(s, project, a)).name

    assert await _in_tx(project, check) == (0, 'a')


@pytest.mark.asyncio
async def test_load_registry_preserves_ids_timestamps_and_the_counter(project) -> None:
    t0 = dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.UTC)
    t1 = dt.datetime(2026, 2, 3, 4, 5, 6, tzinfo=dt.UTC)
    loads = [
        ClassLoad(class_id=7, name='Car', class_group='veh', hotkey_letter='c', added_at=t0),
        ClassLoad(
            class_id=3,
            name='Truck',
            deprecated=True,
            merged_into=7,
            notes='n',
            added_at=t0,
            updated_at=t1,
        ),
        ClassLoad(class_id=9, name='truck', added_at=t1),  # reuses a deprecated name
    ]

    async def run(s):
        await repo.load_registry(
            s,
            project,
            loads,
            revision=12,
            next_class_id=20,
            migrated_from='opensearch_registry_doc',
            migrated_at=t1,
        )

    await _in_tx(project, run)
    rows = await _in_tx(project, lambda s: repo.read_registry(s, project))
    assert (rows.revision, rows.next_class_id) == (12, 20)
    by_id = {c.class_id: c for c in rows.classes}
    assert sorted(by_id) == [3, 7, 9]
    assert by_id[3].merged_into == 7
    assert by_id[3].deprecated
    assert by_id[3].updated_at == t1
    assert by_id[7].added_at == t0
    assert by_id[7].hotkey_letter == 'c'
    assert by_id[9].name_key == 'truck'
    # the normal write path continues from the loaded counter
    new = await _in_tx(project, lambda s: repo.add_class(s, project, 'van'))
    assert new.class_id == 20
    found = await _in_tx(project, lambda s: repo.find_by_name(s, project, 'TRUCK'))
    assert found.class_id == 9


@pytest.mark.asyncio
async def test_load_registry_raises_on_a_conflict_and_on_a_second_load(project) -> None:
    async def twice(s):
        await repo.load_registry(s, project, [ClassLoad(class_id=0, name='a')], revision=1)
        with pytest.raises(ClassRegistryError, match='already has'):
            await repo.load_registry(s, project, [ClassLoad(class_id=0, name='a')], revision=1)

    await _in_tx(project, twice)


@pytest.mark.asyncio
async def test_load_registry_refuses_two_active_classes_with_one_name_key(pg, pg_env) -> None:
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        project = make_project(conn, f'p{uuid.uuid4().hex[:8]}')
    loads = [ClassLoad(class_id=0, name='Car'), ClassLoad(class_id=1, name='car')]

    async def load(s):
        await repo.load_registry(s, project, loads, revision=1)

    with pytest.raises(Exception, match='uq_project_classes_active_name'):
        await _in_tx(project, load)


@pytest.mark.asyncio
async def test_no_cap_on_the_number_of_classes(project) -> None:
    loads = [ClassLoad(class_id=i, name=f'class {i}') for i in range(10_001)]

    async def run(s):
        await repo.load_registry(s, project, loads, revision=1)
        return await repo.add_class(s, project, 'one more')

    added = await _in_tx(project, run)
    assert added.class_id == 10_001
    rows = await _in_tx(project, lambda s: repo.read_registry(s, project))
    assert len(rows.classes) == 10_002


@pytest.mark.asyncio
async def test_copy_registry_replaces_the_target_with_the_source(pg, pg_env) -> None:
    from src.db.session import deployment_tx

    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        src = make_project(conn, f'p{uuid.uuid4().hex[:8]}')
        dst = make_project(conn, f'p{uuid.uuid4().hex[:8]}')
    await _in_tx(src, lambda s: repo.add_class(s, src, 'car'))
    await _in_tx(dst, lambda s: repo.add_class(s, dst, 'stale'))
    # the dst class is unreferenced; copying replaces it
    async with async_session_factory()() as s, deployment_tx(s, 'test copy'):
        assert await repo.copy_registry(s, src, dst) == 1
    rows = await _in_tx(dst, lambda s: repo.read_registry(s, dst))
    assert [c.name for c in rows.classes] == ['car']
    assert rows.next_class_id == 1


@pytest.mark.asyncio
async def test_load_registry_fills_the_empty_row_that_project_create_wrote(project) -> None:
    t0 = dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.UTC)

    async def run(s):
        await repo.ensure_registry(s, project)
        await repo.load_registry(
            s,
            project,
            [ClassLoad(class_id=4, name='Car', added_at=t0)],
            revision=9,
            next_class_id=12,
            migrated_from='class_registry_file',
            migrated_at=t0,
        )
        return await repo.read_registry(s, project)

    rows = await _in_tx(project, run)
    assert (rows.revision, rows.next_class_id) == (9, 12)
    assert [(c.class_id, c.name) for c in rows.classes] == [(4, 'Car')]


@pytest.mark.asyncio
async def test_load_registry_refuses_an_empty_row_that_has_been_written_to(project) -> None:
    async def run(s):
        await repo.add_class(s, project, 'a')
        await repo.set_deprecated(s, project, 0, True)  # revision moved; still one class
        with pytest.raises(ClassRegistryError, match='already has'):
            await repo.load_registry(s, project, [ClassLoad(class_id=0, name='b')], revision=1)

    await _in_tx(project, run)
