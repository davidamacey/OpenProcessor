"""``apply`` takes the locks in the plan's order, appends, loads, folds and stores (15a sections 4.1, 7).

Every repository call is replaced by a recorder, so the test reads the order of the steps and what is
handed to the fold, with no database. Two concurrent sessions and the fold against real rows are in
``tests/db/test_storage_apply.py`` (pending the wave run).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.db.repos.fold_load import LoadedItem
from src.db.repos.results import ResultWrite
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


if TYPE_CHECKING:
    import pytest


PROJECT = uuid.UUID('00000000-0000-4000-8000-000000000001')
OP = str(uuid.UUID('00000000-0000-4000-8000-0000000000aa'))
T0 = datetime(2026, 1, 1, tzinfo=UTC)
BIG = (0.1, 0.1, 0.9, 0.9)
SMALL = (0.1, 0.1, 0.2, 0.2)
CONTEXT = FoldContext(FoldSettings('primary', DEFAULT_PRECEDENCE), lambda _readings: None)


def _loaded(
    key: int, box: tuple[float, float, float, float], *, claims: tuple[Any, ...] = ()
) -> LoadedItem:
    crop = f'{key:032x}'
    detection = DetectionResult(
        'detector', 'primary', '1', box, T0, class_name='car', analyser_id=1
    )
    return LoadedItem(
        ItemInputs(
            item=ItemRow(crop_id=crop, photo_key=7, created_at=T0),
            frame=1,
            detections=(detection,),
            claims=claims,
        ),
        {},
    )


class Recorder:
    """Stands in for every repository call ``apply`` makes."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, *, existing: tuple[int, ...] = (1, 2)
    ) -> None:
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
