"""``apply`` takes the locks in the plan's order, appends, loads, folds and stores (15a sections 4.1, 7).

Every repository call is replaced by a recorder, so the test reads the order of the steps and what is
handed to the fold, with no database. Two concurrent sessions and the fold against real rows are in
``tests/db/test_storage_apply.py`` (pending the wave run).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy.dialects import postgresql

from src.db.repos import items as items_repo
from src.db.repos.fold_load import LoadedItem
from src.db.repos.results import (
    RANK_CHANGING_TABLES,
    RANK_NEUTRAL_TABLES,
    RESULT_SPECS,
    ResultWrite,
)
from src.domain import claim_builders
from src.domain.claim_kinds import ActorKind, ClaimKind
from src.domain.claims import ClaimDraft, commit
from src.domain.fold_state import FoldSettings
from src.domain.results import (
    DEFAULT_PRECEDENCE,
    DetectionResult,
    ItemInputs,
    ItemRow,
    RegistryClass,
    RegistrySnapshot,
)
from src.services.record import apply as apply_module
from src.services.record.apply import FoldContext, apply


PROJECT = uuid.UUID('00000000-0000-4000-8000-000000000001')
OP = str(uuid.UUID('00000000-0000-4000-8000-0000000000aa'))
T0 = datetime(2026, 1, 1, tzinfo=UTC)
BIG = (0.1, 0.1, 0.9, 0.9)
SMALL = (0.1, 0.1, 0.2, 0.2)
CONTEXT = FoldContext(FoldSettings('primary', DEFAULT_PRECEDENCE), lambda _readings: None)


def _loaded(
    key: int,
    box: tuple[float, float, float, float],
    *,
    claims: tuple[Any, ...] = (),
    photo: int = 7,
) -> LoadedItem:
    crop = f'{key:032x}'
    detection = DetectionResult(
        'detector', 'primary', '1', box, T0, class_name='car', analyser_id=1
    )
    return LoadedItem(
        ItemInputs(
            item=ItemRow(crop_id=crop, photo_key=photo, created_at=T0),
            frame=1,
            detections=(detection,),
            claims=claims,
        ),
        {},
    )


class Recorder:
    """Stands in for every repository call ``apply`` makes."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        existing: tuple[int, ...] = (1, 2),
        ranks: dict[int, int] | None = None,
    ) -> None:
        self.ranks = ranks or {}  # the stored state rows: none, an item new to the fold
        self.calls: list[str] = []
        self.locked_with: list[list[int]] = []
        self.states: dict[int, Any] = {}
        self.deleted: list[int] = []
        self.loaded: dict[int, LoadedItem] = {1: _loaded(1, SMALL), 2: _loaded(2, BIG)}
        self.existing = existing
        for name, fn in {
            'photo_keys_of_items': self.photo_keys_of_items,
            'item_keys_of_photos': self.item_keys_of_photos,
            'lock_items': self.lock_items,
            'append_claims': self.append_claims,
            'upsert_results': self.upsert_results,
            'load_item_inputs': self.load_item_inputs,
            'load_registry': self.load_registry,
            'upsert_item_states': self.upsert_item_states,
            'upsert_box_states': self.upsert_box_states,
            'delete_states': self.delete_states,
            'load_ranks': self.load_ranks,
        }.items():
            monkeypatch.setattr(apply_module, name, fn)

    async def photo_keys_of_items(self, _s: Any, keys: Any) -> list[int]:
        self.calls.append('photos_of_items')
        return [7]

    async def item_keys_of_photos(self, _s: Any, _photos: Any) -> list[int]:
        self.calls.append('items_of_photos')
        return [1, 2]

    async def lock_items(self, _s: Any, keys: Any) -> list[int]:
        self.calls.append('lock_items')
        self.locked_with.append(list(keys))
        return [k for k in keys if k in self.existing]

    async def append_claims(self, _s: Any, _p: Any, drafts: Any) -> list[int]:
        self.calls.append('append_claims')
        return list(range(1, len(drafts) + 1))

    async def upsert_results(self, _s: Any, _p: Any, writes: Any) -> int:
        self.calls.append('upsert_results')
        return len(writes)

    async def load_item_inputs(
        self, _s: Any, keys: Any, *, text_consensus: Any
    ) -> dict[int, LoadedItem]:
        self.calls.append('load_item_inputs')
        return {k: self.loaded[k] for k in keys}

    async def load_registry(self, _s: Any, _p: Any) -> RegistrySnapshot:
        self.calls.append('load_registry')
        return RegistrySnapshot((RegistryClass(1, 'car'),))

    async def upsert_item_states(self, _s: Any, _p: Any, folds: Any) -> int:
        self.calls.append('upsert_item_states')
        self.states = dict(folds)
        return len(folds)

    async def upsert_box_states(self, _s: Any, _p: Any, folds: Any, _keys: Any) -> int:
        self.calls.append('upsert_box_states')
        return 0

    async def load_ranks(self, _s: Any, keys: Any) -> dict[int, int]:
        self.calls.append('load_ranks')
        return {k: self.ranks[k] for k in keys if k in self.ranks}

    async def delete_states(self, _s: Any, keys: Any) -> int:
        self.calls.append('delete_states')
        self.deleted = list(keys)
        return len(keys)


def run(**kwargs: Any) -> Any:
    return asyncio.run(apply(object(), PROJECT, context=CONTEXT, **kwargs))  # type: ignore[arg-type]


def _draft() -> Any:
    view = claim_builders.ItemView('ab' * 16, 7, 1, BIG, BIG)
    return claim_builders.label(view, class_name='car', class_id_hint=1, op_id=OP)


def test_the_steps_run_in_the_plans_order(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder(monkeypatch)
    out = run(
        claims=_draft(),
        results=[ResultWrite('detections', [{'item_key': 1}])],
        item_keys=[2, 1],
    )
    assert rec.calls == [
        'photos_of_items', 'items_of_photos', 'lock_items', 'append_claims', 'upsert_results',
        'load_item_inputs', 'load_registry', 'upsert_item_states', 'upsert_box_states',
        'delete_states',
    ]  # fmt: skip
    assert out.appended == (1,)
    assert out.locked == (1, 2)


def test_callers_that_name_the_same_items_in_opposite_orders_lock_the_same_keys_in_the_same_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = Recorder(monkeypatch)
    run(item_keys=[2, 1])
    run(item_keys=[1, 2, 2])
    assert rec.locked_with == [[1, 2], [1, 2]]


def test_the_locks_are_taken_before_anything_is_appended_or_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = Recorder(monkeypatch)
    run(claims=_draft(), item_keys=[1])
    assert rec.calls.index('lock_items') < rec.calls.index('append_claims')
    assert rec.calls.index('lock_items') < rec.calls.index('load_item_inputs')
    assert rec.calls.index('append_claims') < rec.calls.index('load_item_inputs')


def test_nothing_is_appended_or_upserted_when_there_is_nothing_to_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = Recorder(monkeypatch)
    run(item_keys=[1])
    assert 'append_claims' not in rec.calls
    assert 'upsert_results' not in rec.calls
    assert 'load_item_inputs' in rec.calls  # a refold with no new inputs is still a fold


def test_every_item_of_the_touched_photos_is_folded_so_the_ranks_are_right(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = Recorder(monkeypatch)
    out = run(item_keys=[1])  # only the small item was named
    assert out.locked == (1, 2)
    assert {k: f.state.rank for k, f in rec.states.items()} == {1: 2, 2: 1}


def test_an_item_deleted_meanwhile_is_reported_and_the_rest_is_folded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = Recorder(monkeypatch, existing=(2,))
    out = run(item_keys=[1, 2])
    assert out.missing == (1,)
    assert out.locked == (2,)
    assert sorted(rec.states) == [2]


def test_an_item_with_nothing_left_has_its_state_removed_not_its_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = Recorder(monkeypatch)
    gone = _loaded(1, SMALL)
    rec.loaded[1] = LoadedItem(
        ItemInputs(item=gone.inputs.item, frame=1, detections=(), claims=()), {}
    )
    out = run(item_keys=[1])
    assert out.vanished == (1,)
    assert rec.deleted == [1]
    assert sorted(rec.states) == [2]


def test_a_lost_import_claim_is_reported_as_a_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder(monkeypatch)
    view = claim_builders.ItemView(f'{1:032x}', 7, 1, SMALL, SMALL)
    human = claim_builders.label(view, class_name='car', class_id_hint=1, op_id=OP)
    imported = ClaimDraft(
        ActorKind.IMPORT, f'import:{OP}', OP, ClaimKind.CLASS_SET, view.subject(), human[0].payload
    )
    claims = tuple(
        commit(d, claim_id=n, at=T0) for n, d in enumerate([imported, human[0]], start=1)
    )
    rec.loaded[1] = _loaded(1, SMALL, claims=claims)
    out = run(item_keys=[1])
    assert [c.kind for c in out.conflicts] == ['locked_conflict']


# --- the scope of one apply ---------------------------------------------------------------------


def test_every_result_table_is_classified_for_its_effect_on_rank() -> None:
    assert RANK_CHANGING_TABLES.isdisjoint(RANK_NEUTRAL_TABLES)
    assert set(RESULT_SPECS) == RANK_CHANGING_TABLES | RANK_NEUTRAL_TABLES


def _bulk(rec: Recorder, n: int, photos: int) -> tuple[list[int], list[ClaimDraft]]:
    """N items spread over M photos, each with a state row, and one label claim per item."""
    keys = list(range(1, n + 1))
    drafts: list[ClaimDraft] = []
    for k in keys:
        photo = 100 + k % photos
        rec.loaded[k] = _loaded(k, SMALL, photo=photo)
        rec.ranks[k] = 1
        view = claim_builders.ItemView(f'{k:032x}', photo, 1, SMALL, SMALL)
        drafts += claim_builders.label(view, class_name='car', class_id_hint=1, op_id=OP)
    rec.existing = tuple(keys)
    return keys, drafts


def test_a_plain_bulk_label_locks_exactly_the_named_items_and_reads_no_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = Recorder(monkeypatch)
    keys, drafts = _bulk(rec, 12, 4)
    out = run(claims=drafts, item_keys=list(reversed(keys)))
    assert rec.locked_with == [keys]  # one lock statement, exactly N rows, ascending
    assert 'photos_of_items' not in rec.calls
    assert 'items_of_photos' not in rec.calls
    assert out.locked == tuple(keys)
    assert out.ranked_together is False
    assert sorted(rec.states) == keys


def test_the_lock_statement_of_a_plain_bulk_label_names_exactly_the_n_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SQL the real ``lock_items`` sends, read from a recording session."""
    from _fakes import FakeResult, FakeSession, Row

    rec = Recorder(monkeypatch)
    keys, drafts = _bulk(rec, 9, 3)
    session = FakeSession(FakeResult([Row(item_key=k) for k in keys]))

    async def real_lock(_s: Any, wanted: Any) -> list[int]:
        return await items_repo.lock_items(session, wanted)

    monkeypatch.setattr(apply_module, 'lock_items', real_lock)
    run(claims=drafts, item_keys=keys)
    assert len(session.calls) == 1
    assert session.sql().endswith('FOR UPDATE')
    bound = next(iter(session.calls[0][0].compile(dialect=postgresql.dialect()).params.values()))
    assert list(bound) == keys


def test_an_item_with_no_state_row_is_ranked_with_its_photo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = Recorder(monkeypatch, ranks={1: 2})  # item 2 has none
    out = run(claims=_draft(), item_keys=[1, 2])
    assert out.ranked_together is True
    assert 'items_of_photos' in rec.calls


@pytest.mark.parametrize(
    'make',
    [
        pytest.param(lambda v: claim_builders.exclude(v, reason='blurry', op_id=OP), id='exclude'),
        pytest.param(lambda v: claim_builders.restore(v, op_id=OP), id='restore'),
        pytest.param(lambda v: [claim_builders.withdraw(5, v, op_id=OP)], id='withdraw'),
    ],
)
def test_a_claim_that_can_change_rank_takes_the_photo_scope(
    monkeypatch: pytest.MonkeyPatch, make: Any
) -> None:
    rec = Recorder(monkeypatch, ranks={1: 2, 2: 1})
    view = claim_builders.ItemView(f'{1:032x}', 7, 1, SMALL, SMALL)
    out = run(claims=make(view), item_keys=[1])
    assert out.ranked_together is True
    assert out.locked == (1, 2)
    assert rec.locked_with == [[1, 2]]


def test_a_detection_result_or_a_photo_scope_asked_for_takes_the_photo_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = Recorder(monkeypatch, ranks={1: 2, 2: 1})
    detections = [ResultWrite('detections', [{'item_key': 1}])]
    assert run(results=detections, item_keys=[1]).ranked_together is True
    assert run(item_keys=[1], photo_scope=True).ranked_together is True
    neutral = [ResultWrite('class_opinions', [{'item_key': 1}])]
    assert run(results=neutral, item_keys=[1]).ranked_together is False
    assert rec.locked_with == [[1, 2], [1, 2], [1]]


def test_the_named_scope_keeps_each_items_stored_rank_and_equals_the_photo_scope_for_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The differential: a plain label folded alone gives the state the photo-wide path gives it."""
    ranks = {1: 2, 2: 1}  # the small item is second, as its area says
    rec = Recorder(monkeypatch, ranks=dict(ranks))
    view = claim_builders.ItemView(f'{1:032x}', 7, 1, SMALL, SMALL)
    label = claim_builders.label(view, class_name='car', class_id_hint=1, op_id=OP)
    rec.loaded[1] = _loaded(
        1, SMALL, claims=tuple(commit(d, claim_id=n, at=T0) for n, d in enumerate(label, 1))
    )
    named = run(claims=label, item_keys=[1])
    named_state = rec.states[1].state
    together = run(claims=label, item_keys=[1], photo_scope=True)
    assert (named.ranked_together, together.ranked_together) == (False, True)
    assert named_state == rec.states[1].state
    assert named_state.rank == ranks[1]
    assert named_state.class_validated is True
