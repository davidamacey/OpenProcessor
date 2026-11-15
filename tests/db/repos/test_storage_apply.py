"""``apply`` and the repositories of the storage record against a real PostgreSQL (pending the wave run).

The property under test throughout is the one the plan states: after any sequence of writes, every
``item_state`` row equals a fresh fold of the rows the database holds, and no claim is lost. Two sessions
write the same items at once, in opposite key orders; a duplicate chunk adds nothing; an undo killed
between chunks leaves the state equal to the fold of the live claims; claims read back equal what was
written; 100,000 claims are appended and the rate printed.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any

import psycopg
import pytest
from _research_seed import make_project
from _storage_seed import analyser, detection, item, photo, wire
from sqlalchemy import text

from src.db.repos import (
    analysers as analyser_repo,
    claims as claim_repo,
    photos as photo_repo,
    results as result_repo,
    state as state_repo,
)
from src.db.repos.fold_load import load_item_inputs, load_registry
from src.db.session import async_session_factory, project_tx
from src.domain import claim_builders
from src.domain.claim_kinds import ClaimKind
from src.domain.claim_payloads import ClassSet
from src.domain.claims import ClaimDraft, Subject, SubjectKind
from src.domain.fold import fold_item, rank_items
from src.domain.fold_state import FoldSettings
from src.domain.results import DEFAULT_PRECEDENCE
from src.services.record.apply import FoldContext, apply


if TYPE_CHECKING:
    from fixtures.postgres import PgDatabase

pytestmark = [pytest.mark.db, pytest.mark.asyncio, pytest.mark.usefixtures('bound_async_engine')]

SETTINGS = FoldSettings('primary', DEFAULT_PRECEDENCE)
CONTEXT = FoldContext(SETTINGS, lambda _readings: None)
BOX = (0.1, 0.1, 0.5, 0.5)


class World:
    def __init__(
        self, project: uuid.UUID, photos: list[int], items: list[int], crops: list[str]
    ) -> None:
        self.project = project
        self.photos = photos
        self.items = items
        self.crops = crops

    def view(self, n: int) -> claim_builders.ItemView:
        return claim_builders.ItemView(self.crops[n], self.photos[n], 1, BOX, BOX)


@pytest.fixture
def world(pg: PgDatabase) -> World:
    """A project with a class registry (car, truck) and six photos, each with one detected item."""
    slug = f'apply-{uuid.uuid4().hex[:8]}'
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        project = make_project(conn, slug)
        conn.execute(
            'INSERT INTO class_registries (project_id, next_class_id) VALUES (%s, 2)', (project,)
        )
        for class_id, name in ((0, 'car'), (1, 'truck')):
            conn.execute(
                'INSERT INTO project_classes (project_id, class_id, name, name_key) VALUES (%s, %s, %s, %s)',
                (project, class_id, name, name),
            )
        det = analyser(conn, name='primary')
        photos, items, crops = [], [], []
        for n in range(6):
            ph = photo(conn, project, f'p{n}')
            it = item(conn, project, ph, f'i{n}')
            detection(conn, project, ph, it, det)
            photos.append(ph)
            items.append(it)
            crops.append(wire(project, f'i{n}').hex)
    return World(project, photos, items, crops)


async def tx_apply(w: World, **kwargs: Any) -> Any:
    async with async_session_factory()() as session, project_tx(session, w.project):
        return await apply(session, w.project, context=CONTEXT, **kwargs)


def _label(w: World, n: int, name: str, op: uuid.UUID) -> list[ClaimDraft]:
    return claim_builders.label(w.view(n), class_name=name, class_id_hint=None, op_id=str(op))


async def assert_state_is_a_fresh_fold(w: World, item_keys: list[int]) -> None:
    async with async_session_factory()() as session, project_tx(session, w.project):
        loaded = await load_item_inputs(session, item_keys, text_consensus=lambda _r: None)
        registry = await load_registry(session, w.project)
        folds = {k: fold_item(loaded[k].inputs, registry, SETTINGS) for k in item_keys}
        by_photo: dict[int, list[int]] = {}
        for k, f in folds.items():
            if f is not None:
                by_photo.setdefault(f.state.photo_key, []).append(k)
        for keys in by_photo.values():
            ranked = rank_items([folds[x] for x in keys])  # type: ignore[misc]
            folds.update(dict(zip(keys, ranked, strict=True)))
        stored = {
            r['item_key']: dict(r)
            for r in (
                await session.execute(
                    text('SELECT * FROM item_state WHERE item_key = ANY(:k)'), {'k': item_keys}
                )
            ).mappings()
        }
        for key in item_keys:
            fold = folds[key]
            assert fold is not None
            expected = state_repo.item_state_row(w.project, key, fold.state)
            got = stored[key]
            for column, value in expected.items():
                if column == 'updated_at':
                    continue
                if isinstance(value, list) and value and isinstance(value[0], float):
                    assert got[column] == pytest.approx(value, abs=1e-6), (key, column)
                else:
                    assert got[column] == value, (key, column)


async def test_a_label_is_stored_as_state_and_a_second_apply_writes_no_row(world: World) -> None:
    op = uuid.uuid4()
    first = await tx_apply(world, claims=_label(world, 0, 'car', op), item_keys=[world.items[0]])
    assert first.appended
    assert first.state_rows_written >= 1
    await assert_state_is_a_fresh_fold(world, [world.items[0]])
    again = await tx_apply(world, item_keys=[world.items[0]])
    assert again.appended == ()
    assert again.state_rows_written == 0


async def test_two_sessions_labelling_the_same_items_in_opposite_orders_lose_no_claim_and_never_deadlock(
    world: World,
) -> None:
    keys = world.items[:4]
    for round_ in range(10):
        a, b = uuid.uuid4(), uuid.uuid4()
        forward = [d for n in range(4) for d in _label(world, n, 'car', a)]
        backward = [d for n in reversed(range(4)) for d in _label(world, n, 'truck', b)]
        await asyncio.gather(
            tx_apply(world, claims=forward, item_keys=keys),
            tx_apply(world, claims=backward, item_keys=list(reversed(keys))),
        )
        await assert_state_is_a_fresh_fold(world, keys)
        async with async_session_factory()() as session, project_tx(session, world.project):
            held = (
                await session.execute(
                    text('SELECT count(*) FROM claims WHERE op_id = ANY(:ops)'), {'ops': [a, b]}
                )
            ).scalar_one()
        assert held == 8, f'round {round_}: a claim was lost'


async def test_a_duplicate_chunk_adds_nothing(world: World) -> None:
    op = uuid.uuid4()
    drafts = [d for n in range(3) for d in _label(world, n, 'car', op)]
    first = await tx_apply(world, claims=drafts, item_keys=world.items[:3])
    second = await tx_apply(world, claims=drafts, item_keys=world.items[:3])
    assert len(first.appended) == 3
    assert second.appended == ()
    assert second.state_rows_written == 0
    await assert_state_is_a_fresh_fold(world, world.items[:3])


async def _live(world: World, op: uuid.UUID, after: int = 0, limit: int = 1000) -> list[Any]:
    async with async_session_factory()() as session, project_tx(session, world.project):
        return await claim_repo.live_claims_of_operation(
            session, str(op), after_claim_id=after, limit=limit
        )


async def test_an_undo_killed_between_chunks_leaves_the_state_equal_to_the_fold_of_the_live_claims(
    world: World,
) -> None:
    op, undo = uuid.uuid4(), uuid.uuid4()
    keys = world.items[:4]
    await tx_apply(
        world, claims=[d for n in range(4) for d in _label(world, n, 'car', op)], item_keys=keys
    )
    page = await _live(world, op, limit=2)
    assert len(page) == 2
    drafts = [claim_builders.withdraw(c.claim_id, world.view(0), op_id=str(undo)) for c in page]
    await tx_apply(world, claims=drafts, item_keys=[c.item_key for c in page])
    await assert_state_is_a_fresh_fold(world, keys)

    rest = await _live(world, op)
    assert (
        len(rest) == 2
    )  # the chunk that committed is no longer live: the resume finds only what is left
    dying = [claim_builders.withdraw(c.claim_id, world.view(0), op_id=str(undo)) for c in rest]
    with pytest.raises(RuntimeError, match='killed'):
        await _apply_then_die(world, dying, [c.item_key for c in rest])
    await assert_state_is_a_fresh_fold(world, keys)
    assert len(await _live(world, op)) == 2

    await tx_apply(world, claims=dying, item_keys=[c.item_key for c in rest])
    assert await _live(world, op) == []
    await assert_state_is_a_fresh_fold(world, keys)


async def _apply_then_die(w: World, drafts: list[ClaimDraft], keys: list[int]) -> None:
    """``apply`` inside a transaction that never commits: the process is killed before the commit."""
    async with async_session_factory()() as session, project_tx(session, w.project):
        await apply(session, w.project, claims=drafts, item_keys=keys, context=CONTEXT)
        msg = 'killed before commit'
        raise RuntimeError(msg)


async def test_claims_read_back_equal_what_was_written(world: World) -> None:
    op = uuid.uuid4()
    view = world.view(0)
    drafts = [
        *_label(world, 0, 'car', op),
        *claim_builders.exclude(view, reason='not_an_object', op_id=str(op)),
    ]
    out = await tx_apply(world, claims=drafts, item_keys=[world.items[0]])
    await tx_apply(
        world,
        claims=[claim_builders.withdraw(out.appended[0], view, op_id=str(uuid.uuid4()))],
        item_keys=[world.items[0]],
    )
    async with async_session_factory()() as session, project_tx(session, world.project):
        loaded = (await claim_repo.load_claims_for_items(session, [world.items[0]]))[world.items[0]]
    assert [c.kind for c in loaded] == [ClaimKind.CLASS_SET, ClaimKind.EXCLUDED, ClaimKind.WITHDRAW]
    assert [c.claim_id for c in loaded] == sorted(c.claim_id for c in loaded)
    assert loaded[0].payload == drafts[0].payload
    assert loaded[0].subject.crop_id == world.crops[0]
    assert loaded[0].subject.bbox == pytest.approx(BOX, abs=1e-6)
    assert loaded[2].withdraws_claim_id == out.appended[0]
    assert loaded[0].op_id == str(op)


async def test_results_replace_by_analyser_and_an_unchanged_write_changes_nothing(
    world: World,
) -> None:
    async with async_session_factory()() as session, project_tx(session, world.project):
        det = await analyser_repo.get_or_create_analyser(
            session, kind='detector', name='primary', version='1', params={}
        )
        vlm = await analyser_repo.get_or_create_analyser(
            session, kind='vlm', name='m@1', version='1', params={'pack': 'p'}
        )
        again = await analyser_repo.get_or_create_analyser(
            session, kind='vlm', name='m@1', version='1', params={'pack': 'p'}
        )
        assert again == vlm
        assert det != vlm
    key = world.items[0]
    opinion = {
        'item_key': key,
        'source': 'vlm',
        'analyser_id': vlm,
        'class_name': 'truck',
        'outcome': 'matched',
    }
    empty = {**opinion, 'class_name': None, 'outcome': 'empty'}
    write = result_repo.ResultWrite('class_opinions', [opinion])
    async with async_session_factory()() as session, project_tx(session, world.project):
        assert await result_repo.upsert_results(session, world.project, [write]) == 1
        assert await result_repo.upsert_results(session, world.project, [write]) == 0
        await result_repo.upsert_results(
            session, world.project, [result_repo.ResultWrite('class_opinions', [empty])]
        )
        held = (
            await session.execute(
                text('SELECT class_name FROM class_opinions WHERE item_key = :k'), {'k': key}
            )
        ).scalar_one()
    assert held == 'truck'  # an answer with no class never replaces one that has a class (M17)
    await tx_apply(world, item_keys=[key])
    await assert_state_is_a_fresh_fold(world, [key])


async def test_the_work_list_holds_photos_whose_analyser_is_not_the_current_one(
    world: World,
) -> None:
    async with async_session_factory()() as session, project_tx(session, world.project):
        old = await analyser_repo.get_or_create_analyser(
            session, kind='detector', name='primary', version='1', params={}
        )
        new = await analyser_repo.get_or_create_analyser(
            session, kind='detector', name='primary', version='2', params={}
        )
        await analyser_repo.record_ledger(
            session,
            world.project,
            [analyser_repo.LedgerEntry(p, 'detector', old, 'done') for p in world.photos],
        )
        assert (
            await analyser_repo.photos_needing(
                session, world.project, kind='detector', analyser_id=old
            )
            == []
        )
        todo = await analyser_repo.photos_needing(
            session, world.project, kind='detector', analyser_id=new
        )
        assert todo == sorted(world.photos)
        await analyser_repo.record_ledger(
            session,
            world.project,
            [analyser_repo.LedgerEntry(world.photos[0], 'detector', new, 'done')],
        )
        assert await analyser_repo.photos_needing(
            session, world.project, kind='detector', analyser_id=new
        ) == sorted(world.photos[1:])


async def test_a_photo_that_exists_is_not_stored_twice(world: World) -> None:
    new = photo_repo.NewPhoto(
        wire(world.project, 'p0').hex, b'\x01' * 16, 'root/x.jpg', 1, 2, 3, 1, 'gpu', 'test'
    )
    fresh = photo_repo.NewPhoto(
        wire(world.project, 'new').hex, b'\x02' * 16, 'root/y.jpg', 1, 2, 3, 1, 'gpu', 'test'
    )
    async with async_session_factory()() as session, project_tx(session, world.project):
        stored = await photo_repo.insert_photos(session, world.project, [new, fresh])
        assert list(stored) == [fresh.image_id]
        found = await photo_repo.photo_keys_by_image_id(
            session, world.project, [new.image_id, fresh.image_id]
        )
        assert found[new.image_id] == world.photos[0]
        locked = await photo_repo.lock_photos(session, [stored[fresh.image_id], world.photos[0]])
        assert locked == sorted(locked)


async def test_one_hundred_thousand_claims_are_appended_and_the_rate_is_printed(
    world: World,
) -> None:
    op = uuid.uuid4()
    drafts = [
        ClaimDraft(
            claim_builders.ActorKind.HUMAN, 'api', str(op), ClaimKind.CLASS_SET,
            Subject(SubjectKind.ITEM, photo_key=world.photos[0], crop_id=f'{n:032x}', bbox=BOX, frame=1),
            ClassSet(class_name='car', class_id_hint=None, label_source='human', via='bulk_accept', labeler='human'),
        )
        for n in range(100_000)
    ]  # fmt: skip
    started = time.perf_counter()
    async with async_session_factory()() as session, project_tx(session, world.project):
        ids = await claim_repo.append_claims(session, world.project, drafts)
    seconds = time.perf_counter() - started
    assert len(ids) == 100_000
    print(f'claims appended: {len(ids)} in {seconds:.1f} s = {len(ids) / seconds:,.0f} claims/s')
