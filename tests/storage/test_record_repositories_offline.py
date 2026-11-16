"""The repositories of the storage record read as SQL, with no database.

A fake session records each statement; the tests read what is sent (the shape of the upsert, the lock
mode and order, the number of round trips) and the pure conversions between a claim and its row. What a
server answers is in the ``db`` tests of this change (pending the wave run).
"""

from __future__ import annotations

import asyncio
import struct
import uuid
from datetime import UTC, datetime
from typing import Any

import _storage_plan as plan
import pytest
from _fakes import FakeResult, FakeSession, Row
from sqlalchemy.dialects import postgresql

from src.db.models.analysis import AnalysisLedger
from src.db.models.claims import Claim as ClaimRow
from src.db.models.items import Detection
from src.db.models.scores import ItemProjection
from src.db.repos import analysers, claims, items, photos, results, state, upserts
from src.db.repos.claim_rows import CLAIM_COLUMNS, claim_from_row, draft_row
from src.db.repos.fold_load import load_item_inputs
from src.domain import claim_builders
from src.domain.claims import ClaimDraft, commit
from src.domain.fold import fold_item
from src.domain.fold_state import BoxState, FoldSettings
from src.domain.results import (
    DEFAULT_PRECEDENCE,
    DetectionResult,
    ItemInputs,
    ItemRow,
    RegistryClass,
    RegistrySnapshot,
)
from src.domain.wire_ids import to_wire


PROJECT = uuid.UUID('00000000-0000-4000-8000-000000000001')
CROP = 'ab' * 16
OP = str(uuid.UUID('00000000-0000-4000-8000-0000000000aa'))
BOX = (0.1, 0.1, 0.5, 0.5)
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def f32(values: tuple[float, ...]) -> list[float]:
    return list(struct.unpack(f'{len(values)}f', struct.pack(f'{len(values)}f', *values)))


# --- the upsert statement ---------------------------------------------------------------------


def _detection_row(item: int, analyser: int, **extra: Any) -> dict[str, Any]:
    return {'project_id': PROJECT, 'photo_key': 1, 'item_key': item, 'analyser_id': analyser,
            'bbox': list(BOX), 'class_name': 'car', **extra}  # fmt: skip


def test_an_upsert_updates_only_a_row_that_differs() -> None:
    session = FakeSession()
    run(
        upserts.upsert_distinct(
            session, Detection, [_detection_row(1, 2)], key=('item_key', 'analyser_id')
        )
    )
    sql = session.sql()
    assert 'ON CONFLICT (item_key, analyser_id) DO UPDATE SET' in sql
    set_clause, where = sql.split(' DO UPDATE SET ')[1].split(' WHERE ')
    assert 'item_key' not in set_clause.replace('excluded.item_key', '')
    assert 'bbox = excluded.bbox' in set_clause
    assert 'IS DISTINCT FROM' in where
    for column in ('bbox', 'class_name', 'photo_key', 'project_id'):
        assert f'detections.{column}' in where
        assert f'excluded.{column}' in where


def test_a_column_that_does_not_decide_a_change_is_written_but_not_compared() -> None:
    session = FakeSession()
    rows = [{'item_key': 1, 'project_id': PROJECT, 'analyser_id': 1, 'x': 1.0, 'y': 2.0}]
    run(
        upserts.upsert_distinct(session, ItemProjection, rows, key=('item_key',), no_compare=('y',))
    )
    sql = session.sql()
    set_clause, where = sql.split(' DO UPDATE SET ')[1].split(' WHERE ')
    assert 'y = excluded.y' in set_clause
    assert 'item_projections.y' not in where
    assert 'item_projections.x' in where


def test_the_ledger_stamps_updated_at_only_when_it_changes_another_column() -> None:
    session = FakeSession()
    entry = analysers.LedgerEntry(1, 'detector', 3, 'done')
    run(analysers.record_ledger(session, PROJECT, [entry]))
    set_clause, where = session.sql().split(' DO UPDATE SET ')[1].split(' WHERE ')
    assert 'updated_at = now()' in set_clause
    assert 'updated_at' not in where
    assert 'ON CONFLICT (photo_key, kind)' in session.sql()
    assert AnalysisLedger.__tablename__ in session.sql()


def test_rows_of_one_statement_must_share_their_columns_and_hold_the_key() -> None:
    with pytest.raises(upserts.RowShapeError):
        run(
            upserts.upsert_distinct(
                FakeSession(), Detection, [{'item_key': 1}, {'analyser_id': 1}], key=('item_key',)
            )
        )
    with pytest.raises(upserts.RowShapeError):
        run(upserts.upsert_distinct(FakeSession(), Detection, [{'bbox': [1]}], key=('item_key',)))


def test_a_key_given_twice_in_one_statement_keeps_the_last_row() -> None:
    session = FakeSession()
    rows = [_detection_row(1, 2, class_name='car'), _detection_row(1, 2, class_name='van')]
    run(upserts.upsert_distinct(session, Detection, rows, key=('item_key', 'analyser_id')))
    ((statement, _),) = session.calls
    bound = statement.compile(dialect=postgresql.dialect()).params.values()
    assert 'van' in bound
    assert 'car' not in bound


def test_a_large_batch_is_split_under_the_parameter_limit_and_nothing_is_sent_for_none() -> None:
    session = FakeSession()
    rows = [_detection_row(i, 1) for i in range(12_000)]
    run(upserts.upsert_distinct(session, Detection, rows, key=('item_key', 'analyser_id')))
    assert len(session.calls) > 1
    per_statement = max(
        len(c[0].compile(dialect=postgresql.dialect()).params) for c in session.calls
    )
    assert per_statement < 65_535
    empty = FakeSession()
    assert run(upserts.upsert_distinct(empty, Detection, [], key=('item_key',))) == 0
    assert empty.calls == []


# --- claims ------------------------------------------------------------------------------------


def _view() -> claim_builders.ItemView:
    return claim_builders.ItemView(crop_id=CROP, photo_key=7, frame=1, bbox=BOX, machine_bbox=BOX)


def _drafts() -> list[ClaimDraft]:
    view = _view()
    return [
        *claim_builders.label(view, class_name='car', class_id_hint=1, op_id=OP),
        *claim_builders.exclude(view, reason='not_an_object', op_id=OP),
        *claim_builders.holdout_freeze(view, freeze_id=OP, machine_class=None, op_id=OP),
        claim_builders.withdraw(5, view, op_id=OP),
        *claim_builders.duplicate_verdict(CROP, 'cd' * 16, level='crop', decision='same'),
    ]


def test_every_claim_column_a_query_selects_is_a_column_of_the_table() -> None:
    assert set(CLAIM_COLUMNS) <= {c.name for c in ClaimRow.__table__.columns}


def test_an_insert_row_names_every_column_and_leaves_the_id_and_time_to_the_table() -> None:
    rows = [draft_row(PROJECT, d) for d in _drafts()]
    assert len({tuple(r) for r in rows}) == 1
    assert 'claim_id' not in rows[0]
    assert 'at' not in rows[0]
    assert set(rows[0]) <= {c.name for c in ClaimRow.__table__.columns}


def test_a_claim_survives_its_row_up_to_the_float4_a_box_is_stored_in() -> None:
    for n, draft in enumerate(_drafts(), start=1):
        stored = draft_row(PROJECT, draft)
        for column in ('subject_bbox', 'subject_machine_bbox'):
            if stored[column] is not None:
                stored[column] = f32(tuple(stored[column]))  # what a real[] column hands back
        row = {**stored, 'claim_id': n, 'at': T0}
        loaded = claim_from_row(row)
        expected = commit(draft, claim_id=n, at=T0)
        assert loaded.kind is expected.kind
        assert loaded.payload == expected.payload
        assert loaded.subject.crop_id == expected.subject.crop_id
        assert loaded.subject.photo_key == expected.subject.photo_key
        assert loaded.op_id == expected.op_id
        if expected.subject.bbox is not None:
            assert loaded.subject.bbox == pytest.approx(expected.subject.bbox, abs=1e-6)
        assert loaded.withdraws_claim_id == expected.withdraws_claim_id


def test_a_withdraw_row_has_no_photo_so_it_is_found_through_its_target() -> None:
    row = draft_row(PROJECT, claim_builders.withdraw(5, _view(), op_id=OP))
    assert row['subject_kind'] == 'claim'
    assert row['photo_key'] is None
    assert row['withdraws_claim_id'] == 5


def test_appending_is_insert_on_conflict_do_nothing_with_no_target() -> None:
    session = FakeSession(FakeResult([Row(claim_id=9), Row(claim_id=4)]))
    ids = run(claims.append_claims(session, PROJECT, _drafts()[:2]))
    assert ids == [4, 9]
    sql = session.sql()
    assert sql.startswith('INSERT INTO claims')
    assert 'ON CONFLICT DO NOTHING' in sql
    assert 'RETURNING claims.claim_id' in sql


def test_nothing_is_sent_to_append_no_claims() -> None:
    session = FakeSession()
    assert run(claims.append_claims(session, PROJECT, [])) == []
    assert session.calls == []


def test_the_claims_of_many_items_are_one_statement_with_their_withdraws() -> None:
    withdraw_row = {
        **draft_row(PROJECT, claim_builders.withdraw(1, _view(), op_id=OP)),
        'claim_id': 2,
        'at': T0,
    }
    body_row = {**draft_row(PROJECT, _drafts()[0]), 'claim_id': 1, 'at': T0}
    session = FakeSession(
        FakeResult([Row(item_key=3, **body_row), Row(item_key=3, **withdraw_row)])
    )
    loaded = run(claims.load_claims_for_items(session, [5, 3, 5]))
    assert len(session.calls) == 1
    assert session.calls[0][1] == {'keys': [3, 5]}
    sql = ' '.join(str(session.calls[0][0]).split())
    assert 'UNION ALL' in sql
    assert 'w.withdraws_claim_id = b.claim_id' in sql
    assert 'ORDER BY item_key, claim_id' in sql
    assert [c.claim_id for c in loaded[3]] == [1, 2]
    assert loaded[5] == []


def test_the_undo_walk_pages_an_operations_live_claims_by_id() -> None:
    session = FakeSession()
    run(claims.live_claims_of_operation(session, OP, after_claim_id=40, limit=1000))
    sql = ' '.join(str(session.calls[0][0]).split())
    assert 'c.op_id = :op' in sql
    assert "c.kind <> 'withdraw'" in sql
    assert 'NOT EXISTS' in sql
    assert 'ORDER BY c.claim_id' in sql
    assert session.calls[0][1]['after'] == 40


# --- photos, items, locks ----------------------------------------------------------------------


def test_a_photo_insert_answers_only_the_photos_it_stored() -> None:
    new = photos.NewPhoto(CROP, b'\x00' * 16, 'root/a.jpg', 10, 4, 3, 1, 'gpu', 'upload')
    session = FakeSession(FakeResult([Row(photo_key=11, image_id=uuid.UUID(CROP))]))
    stored = run(photos.insert_photos(session, PROJECT, [new]))
    assert stored == {CROP: 11}
    sql = session.sql()
    assert 'ON CONFLICT (project_id, image_id) DO NOTHING' in sql
    assert 'RETURNING photos.photo_key, photos.image_id' in sql


def test_the_photo_lock_is_for_no_key_update_in_key_order() -> None:
    session = FakeSession(FakeResult([Row(photo_key=2), Row(photo_key=9)]))
    assert run(photos.lock_photos(session, [9, 2, 9])) == [2, 9]
    sql = session.sql()
    assert sql.endswith('FOR NO KEY UPDATE')
    assert 'ORDER BY photos.photo_key' in sql
    assert session.calls[0][0].compile(dialect=postgresql.dialect()).params


def test_the_item_lock_is_for_update_in_ascending_key_order_whatever_the_callers_order() -> None:
    sessions = []
    for order in ([9, 2, 5], [5, 9, 2]):
        s = FakeSession(FakeResult([Row(item_key=2), Row(item_key=5), Row(item_key=9)]))
        run(items.lock_items(s, order))
        sessions.append(s)
    for s in sessions:
        sql = s.sql()
        assert 'item_key = ANY' in sql
        assert 'ORDER BY items.item_key' in sql
        assert sql.endswith('FOR UPDATE')
    # the same statement and the same sorted array: two writers over the same items take the same locks in the same order
    arrays = [
        list(next(iter(s.calls[0][0].compile(dialect=postgresql.dialect()).params.values())))
        for s in sessions
    ]
    assert arrays == [[2, 5, 9], [2, 5, 9]]


def test_no_lock_is_taken_for_no_keys() -> None:
    s = FakeSession()
    assert run(items.lock_items(s, [])) == []
    assert s.calls == []


# --- analysers ---------------------------------------------------------------------------------


def test_a_configuration_hashes_the_same_in_any_key_order() -> None:
    a = analysers.config_sha256({'prompt_pack': 'p', 'revision': 3})
    assert a == analysers.config_sha256({'revision': 3, 'prompt_pack': 'p'})
    assert a != analysers.config_sha256({'prompt_pack': 'p', 'revision': 4})
    assert len(a) == 64


def test_an_existing_analyser_is_found_not_duplicated() -> None:
    session = FakeSession(FakeResult([]), FakeResult([7]))
    got = run(
        analysers.get_or_create_analyser(
            session, kind='vlm', name='m@1', version='1', params={'a': 1}
        )
    )
    assert got == 7
    assert 'ON CONFLICT ON CONSTRAINT uq_analysers_identity DO NOTHING' in session.sql(0)
    assert 'WHERE analysers.kind' in session.sql(1)


def test_the_work_list_is_an_equality_on_the_current_analyser() -> None:
    session = FakeSession()
    run(analysers.photos_needing(session, PROJECT, kind='detector', analyser_id=3))
    sql = ' '.join(str(session.calls[0][0]).split())
    assert 'l.analyser_id <> :analyser' in sql
    assert "l.state <> 'done'" in sql
    assert 'l.photo_key IS NULL' in sql


# --- results -----------------------------------------------------------------------------------


def test_results_are_written_parents_first_and_an_unknown_table_is_refused() -> None:
    session = FakeSession(FakeResult(), FakeResult([Row(item_key=1, box_id='b1', box_key=50)]))
    writes = [
        results.ResultWrite(
            'box_texts',
            [{'item_key': 1, 'box_id': 'b1', 'reader': 'ocr', 'analyser_id': 2, 'text': 'x'}],
        ),
        results.ResultWrite(
            'region_boxes',
            [
                {
                    'item_key': 1,
                    'box_id': 'b1',
                    'analyser_id': 2,
                    'region_seq': 1,
                    'bbox': list(BOX),
                    'state': 'proposed',
                }
            ],
        ),
    ]
    run(results.upsert_results(session, PROJECT, writes))
    tables = [
        session.sql(i).split()[2]
        for i in range(len(session.calls))
        if session.sql(i).startswith('INSERT')
    ]
    assert tables == ['region_boxes', 'box_texts']
    with pytest.raises(results.ResultRowError):
        run(
            results.upsert_results(
                FakeSession(), PROJECT, [results.ResultWrite('claims', [{'a': 1}])]
            )
        )


def test_a_reading_of_a_box_that_does_not_exist_is_a_writer_bug() -> None:
    session = FakeSession(FakeResult([]))
    write = results.ResultWrite(
        'box_texts', [{'item_key': 1, 'box_id': 'zz', 'reader': 'ocr', 'analyser_id': 2}]
    )
    with pytest.raises(results.ResultRowError):
        run(results.upsert_results(session, PROJECT, [write]))


def test_a_vlm_answer_without_a_class_never_replaces_one_with_a_class() -> None:
    session = FakeSession()
    row = {
        'item_key': 1,
        'project_id': PROJECT,
        'source': 'vlm',
        'analyser_id': 2,
        'class_name': None,
        'outcome': 'empty',
    }
    run(results.upsert_results(session, PROJECT, [results.ResultWrite('class_opinions', [row])]))
    where = session.sql().split(' WHERE ')[1]
    assert '(excluded.class_name IS NOT NULL OR class_opinions.class_name IS NULL)' in where
    assert 'IS DISTINCT FROM' in where


def test_the_result_tables_are_all_planned_tables_with_a_conflict_key_that_exists() -> None:
    for name, spec in results.RESULT_SPECS.items():
        assert name in plan.TABLES
        assert set(spec.key) <= set(plan.COLUMNS[name])


# --- state -------------------------------------------------------------------------------------


def _fold() -> Any:
    detection = DetectionResult(
        'detector', 'primary', '1', BOX, T0, class_name='car', analyser_id=1
    )
    inputs = ItemInputs(
        item=ItemRow(crop_id=CROP, photo_key=7, created_at=T0), frame=1, detections=(detection,),
        claims=tuple(
            commit(d, claim_id=n, at=T0) for n, d in enumerate(
                claim_builders.label(_view(), class_name='car', class_id_hint=1, op_id=OP), start=1)
        ),
    )  # fmt: skip
    registry = RegistrySnapshot((RegistryClass(1, 'car'),))
    fold = fold_item(inputs, registry, FoldSettings('primary', DEFAULT_PRECEDENCE))
    assert fold is not None
    return fold


def test_the_item_state_row_is_exactly_the_planned_columns() -> None:
    row = state.item_state_row(PROJECT, 4, _fold().state)
    assert list(row) == list(plan.COLUMNS['item_state'])
    assert row['crop_id'] == uuid.UUID(CROP)
    assert row['class_op_id'] == uuid.UUID(OP)
    assert row['class_validated'] is True


def test_the_box_state_row_is_the_planned_columns_without_the_identity_key() -> None:
    box = BoxState('b1', BOX, 'proposed', 1, detected_at='2026-01-01T00:00:00+00:00')
    row = state.box_state_row(PROJECT, 4, box, {'b1': 50})
    assert list(row) == [c for c in plan.COLUMNS['box_state'] if c != 'box_row_key']
    assert row['box_key'] == 50
    claimed = state.box_state_row(
        PROJECT, 4, BoxState('b2', BOX, 'accepted', 2, from_claims_only=True), {'b2': 51}
    )
    assert claimed['box_key'] is None


def test_the_item_rows_are_written_only_where_they_differ_and_updated_at_does_not_decide() -> None:
    session = FakeSession()
    run(state.upsert_item_states(session, PROJECT, {4: _fold()}))
    set_clause, where = session.sql().split(' DO UPDATE SET ')[1].split(' WHERE ')
    assert 'updated_at = excluded.updated_at' in set_clause
    assert 'item_state.updated_at' not in where
    assert 'item_state.class_validated' in where
    assert 'ON CONFLICT (item_key)' in session.sql()


def test_boxes_the_fold_no_longer_shows_are_deleted_in_the_same_pass() -> None:
    session = FakeSession()
    run(state.upsert_box_states(session, PROJECT, {4: _fold()}, {4: {}}))
    assert any(
        'DELETE FROM box_state' in str(c[0]) and 'NOT EXISTS' in str(c[0]) for c in session.calls
    )


# --- what the fold reads -----------------------------------------------------------------------


@pytest.mark.parametrize('n_items', [1, 50])
def test_the_fold_inputs_cost_one_query_per_table_family_however_many_items(n_items: int) -> None:
    keys = list(range(1, n_items + 1))
    item_rows = [Row(item_key=k, crop_id=uuid.UUID(f'{k:032x}'), photo_key=1, created_at=T0, box_seq=0,
                     geometry_seq=0, region_seq=0, frame=1) for k in keys]  # fmt: skip
    session = FakeSession(FakeResult(item_rows))
    loaded = run(load_item_inputs(session, reversed(keys), text_consensus=lambda _r: None))
    # items, detections, class opinions, region results, boxes, readings, claims
    assert len(session.calls) == 7
    assert sorted(loaded) == keys
    assert all(item.inputs.claims == () for item in loaded.values())
    assert loaded[1].inputs.item.crop_id == to_wire(uuid.UUID(f'{1:032x}'))


def test_no_item_no_query() -> None:
    session = FakeSession()
    assert run(load_item_inputs(session, [], text_consensus=lambda _r: None)) == {}
    assert session.calls == []
