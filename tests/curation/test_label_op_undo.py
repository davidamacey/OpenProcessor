"""Undoing a label operation by id: per item, later edits win, bounded and resumable."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from curation.label_ops_env import build_env, proposal
from src.clients.occ import OCC_BULK_PAGE_SIZE
from src.domain.history import AUDIT_VERDICT_FIELDS, CLASS_STATE_FIELDS
from src.services.curation.label_ops import store


if TYPE_CHECKING:
    from curation.label_ops_env import OpsEnv

COUNT_KEYS = (
    'restored',
    'already_undone',
    'kept_changed',
    'kept_holdout',
    'kept_stale_class',
    'conflicts',
)
STATE_FIELDS = (*CLASS_STATE_FIELDS, *AUDIT_VERDICT_FIELDS)


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, n: int = 10, **extra: Any) -> OpsEnv:
    return build_env(
        monkeypatch, tmp_path, {f'c{i:05d}': proposal(f'c{i:05d}', **extra) for i in range(n)}
    )


def _ids(*numbers: int) -> list[str]:
    return [f'c{i:05d}' for i in numbers]


def _state(doc: dict[str, Any]) -> dict[str, Any]:
    state = {f: doc.get(f) for f in STATE_FIELDS}
    state['class_validated'] = bool(state['class_validated'])
    return state


def _undo_ok(env: OpsEnv, op_id: str, *, dry_run: bool = False) -> dict[str, Any]:
    resp = env.undo(op_id, dry_run=dry_run)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['matched'] == sum(body[k] for k in COUNT_KEYS), 'the counts partition matched'
    return body


def test_exact_restore_of_a_12000_item_selection_op(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audit = {
        'audit_sample': True,
        'detector_class_name': 'thing',
        'audit_label_name': 'thing',
    }
    docs = {
        f'c{i:05d}': proposal(f'c{i:05d}', **(audit if i % 10 == 0 else {})) for i in range(12_000)
    }
    env = build_env(monkeypatch, tmp_path, docs)
    before = {k: _state(v) for k, v in env.docs().items()}

    op = env.label('widget', selection={'filter': {'class_names': ['thing']}})
    assert op['updated'] == 12_000, 'more than the old 5,000-id undo cap'
    sampled = env.docs()['c00000']
    assert sampled['audit_outcome'] is not None, 'the label stamped a verdict'
    assert env.docs()['c00001']['class_validated'] is True

    body = _undo_ok(env, op['op_id'])
    assert body['restored'] == 12_000
    assert body['kept_changed'] == body['conflicts'] == 0
    assert body['members_known'] is True
    after = {k: _state(v) for k, v in env.docs().items()}
    assert after == before, 'every item is back exactly, audit verdict included'


def test_a_human_edit_after_the_op_is_kept_changed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    op = env.label('widget', crop_ids=_ids(*range(10)))
    edit = env.client.put(
        f'{env.prefix}/crops/{_ids(3)[0]}/label', json={'class_id': env.class_ids['gizmo']}
    )
    assert edit.status_code == 200
    body = _undo_ok(env, op['op_id'])
    assert (body['restored'], body['kept_changed']) == (9, 1)
    edited = env.docs()[_ids(3)[0]]
    assert (edited['class_name'], edited['class_validated']) == ('gizmo', True)
    assert env.docs()[_ids(4)[0]]['class_validated'] is False


def test_ops_undo_in_any_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = _env(monkeypatch, tmp_path, n=100)
    before = {k: _state(v) for k, v in env.docs().items()}
    a = env.label('widget', crop_ids=_ids(*range(100)))['op_id']
    b = env.label('gadget', crop_ids=_ids(*range(50, 100)))['op_id']

    first = _undo_ok(env, a)
    assert (first['restored'], first['kept_changed']) == (50, 50)
    listed = {o['op_id']: o for o in env.list_ops().json()['items']}
    assert listed[a]['current'] == 0
    assert listed[a]['undoable'] is False
    assert listed[a]['undone'] is True
    assert listed[a]['last_undo']['restored'] == 50

    second = _undo_ok(env, b)
    assert (second['restored'], second['kept_changed']) == (50, 0)
    listed = {o['op_id']: o for o in env.list_ops().json()['items']}
    assert listed[a]['current'] == 50, 'undoing the later op hands its items back to the first'
    assert listed[a]['undoable'] is True

    third = _undo_ok(env, a)
    assert (third['restored'], third['already_undone'], third['kept_changed']) == (50, 50, 0)
    assert {k: _state(v) for k, v in env.docs().items()} == before


def test_items_undone_one_by_one_count_as_already_undone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    op = env.label('widget', crop_ids=_ids(0, 1, 2, 3, 4, 5))
    for crop_id in _ids(0, 1):
        assert env.client.post(f'{env.prefix}/crops/{crop_id}/label/undo').status_code == 200
    # A machine writer left its own entry on top of an item's history without touching the
    # label: the stack still ends at the operation, so the item is restored.
    top = env.docs()[_ids(5)[0]]
    top['class_id_history'].append({'writer': 'class_merge', 'class_id': 9, 'at': 'later'})
    body = _undo_ok(env, op['op_id'])
    assert (body['restored'], body['already_undone'], body['kept_changed']) == (4, 2, 0)
    assert top['class_validated'] is False


def test_holdout_items_are_kept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = _env(monkeypatch, tmp_path)
    env.docs()[_ids(4)[0]]['test_holdout'] = True  # frozen before the op: labelled by id
    op = env.label('widget', crop_ids=_ids(0, 1, 2, 3, 4))
    assert env.docs()[_ids(4)[0]]['class_validated'] is True
    env.docs()[_ids(1)[0]]['test_holdout'] = True  # frozen after the op
    writes = env.fake.write_calls

    body = _undo_ok(env, op['op_id'], dry_run=True)
    assert (body['restored'], body['kept_holdout']) == (3, 2)
    assert env.fake.write_calls == writes
    body = _undo_ok(env, op['op_id'])
    assert (body['restored'], body['kept_holdout']) == (3, 2)
    for crop_id in _ids(1, 4):
        assert env.docs()[crop_id]['class_validated'] is True, 'the holdout is never touched'


def test_excluded_after_the_op_is_kept_and_before_the_op_is_restored_parked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    before_op = _ids(2)[0]
    env.docs()[before_op].update(class_excluded=True, excluded_at='2026-01-01T00:00:00+00:00')
    op = env.label('widget', crop_ids=_ids(0, 1, 2))
    after_op = _ids(1)[0]
    env.docs()[after_op].update(class_excluded=True, excluded_at='2999-01-01T00:00:00+00:00')

    body = _undo_ok(env, op['op_id'])
    assert (body['restored'], body['kept_changed']) == (2, 1)
    kept = env.docs()[after_op]
    assert kept['class_name'] == 'widget', 'excluded after the op: left as it is'
    parked = env.docs()[before_op]
    assert parked['class_excluded'] is True, 'still excluded'
    assert parked['excluded_prior_class_validated'] is False
    assert parked['excluded_prior_cluster_id'] == 10_003, 'un-exclude brings the old cluster back'


def test_merged_or_renamed_class_is_kept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = _env(monkeypatch, tmp_path, n=6)
    gizmo = env.class_ids['gizmo']
    for crop_id in _ids(3, 4, 5):  # carried a machine class before the op
        env.docs()[crop_id].update(class_id=gizmo, class_name='gizmo', class_source='vlm')
    op = env.label('widget', crop_ids=_ids(*range(6)))
    # A class merge rewrote item 1 to the merge target.
    env.docs()[_ids(1)[0]].update(class_id=env.class_ids['gadget'], class_name='gadget')
    env.registry.rename_class(gizmo, 'gizmo_renamed')
    writes = env.fake.write_calls

    dry = _undo_ok(env, op['op_id'], dry_run=True)
    assert (dry['restored'], dry['kept_changed'], dry['kept_stale_class']) == (2, 1, 3)
    assert env.fake.write_calls == writes
    body = _undo_ok(env, op['op_id'])
    assert (body['restored'], body['kept_changed'], body['kept_stale_class']) == (2, 1, 3)
    for crop_id in _ids(3, 4, 5):
        assert env.docs()[crop_id]['class_name'] == 'widget', 'no resurrected old name'
    assert env.docs()[_ids(1)[0]]['class_name'] == 'gadget'
    assert env.docs()[_ids(0)[0]]['class_validated'] is False


def test_second_undo_is_409_already_undone_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    op = env.label('widget', crop_ids=_ids(0, 1, 2))
    _undo_ok(env, op['op_id'])
    writes = env.fake.write_calls
    snapshot = copy.deepcopy(env.docs())
    for dry_run in (False, True):
        resp = env.undo(op['op_id'], dry_run=dry_run)
        assert resp.status_code == 409, resp.text
        assert resp.json()['detail']['error'] == 'already_undone'
    assert env.fake.write_calls == writes
    assert env.docs() == snapshot


def test_occ_conflict_is_reported_not_swallowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    op = env.label('widget', crop_ids=_ids(*range(5)))
    stuck = _ids(3)[0]
    env.fake.conflict_ids = {stuck}
    body = _undo_ok(env, op['op_id'])
    assert (body['restored'], body['conflicts']) == (4, 1)
    assert body['conflict_ids'] == [stuck]
    assert env.docs()[stuck]['label_op_id'] == op['op_id'], 'still tagged'
    listed = env.list_ops().json()['items'][0]
    assert (listed['current'], listed['undoable']) == (1, True)

    env.fake.conflict_ids = set()
    again = _undo_ok(env, op['op_id'])
    assert (again['restored'], again['already_undone'], again['conflicts']) == (1, 4, 0)
    assert env.docs()[stuck]['class_validated'] is False


def test_dry_run_writes_nothing_and_predicts_the_real_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    op = env.label('widget', crop_ids=_ids(*range(8)))
    env.client.put(
        f'{env.prefix}/crops/{_ids(2)[0]}/label', json={'class_id': env.class_ids['gizmo']}
    )
    writes = env.fake.write_calls
    snapshot = copy.deepcopy(env.docs())
    state_before = (env.state() / op['op_id'] / 'state.json').read_text()

    default = env.undo(op['op_id'], dry_run=None)  # the default is a dry run
    assert default.status_code == 200
    assert default.json()['dry_run'] is True
    dry = _undo_ok(env, op['op_id'], dry_run=True)
    assert env.fake.write_calls == writes
    assert env.docs() == snapshot
    assert (env.state() / op['op_id'] / 'state.json').read_text() == state_before

    real = _undo_ok(env, op['op_id'])
    assert real['dry_run'] is False
    for key in COUNT_KEYS:
        assert real[key] == dry[key], key


def test_undo_keeps_one_request_in_flight(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    n = 1_300
    env = _env(monkeypatch, tmp_path, n=n)
    op = env.label('widget', crop_ids=_ids(*range(n)))
    env.fake.bulk_sizes.clear()
    env.fake.max_in_flight = 0
    updates = env.fake.update_calls

    body = _undo_ok(env, op['op_id'])
    assert body['restored'] == n
    assert env.fake.max_in_flight == 1
    assert env.fake.update_calls == updates, 'no single-document updates'
    assert max(env.fake.bulk_sizes) <= OCC_BULK_PAGE_SIZE
    assert len(env.fake.bulk_sizes) == math.ceil(n / OCC_BULK_PAGE_SIZE)


def test_a_second_undo_in_the_project_is_409_busy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    op = env.label('widget', crop_ids=_ids(0, 1))
    writes = env.fake.write_calls
    with store.undo_lock() as got:
        assert got
        resp = env.undo(op['op_id'], dry_run=False)
        assert resp.status_code == 409
        assert resp.json()['detail']['error'] == 'label_op_busy'
        assert env.undo(op['op_id'], dry_run=True).status_code == 200, 'a dry run takes no lock'
    assert env.fake.write_calls == writes
    assert env.undo(op['op_id'], dry_run=False).status_code == 200


def test_undo_while_writing_is_409_and_an_interrupted_op_is_undoable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    op = env.label('widget', crop_ids=_ids(0, 1))
    with store.write_lock(op['op_id']) as got:
        assert got
        resp = env.undo(op['op_id'], dry_run=False)
        assert resp.status_code == 409
        assert resp.json()['detail']['error'] == 'label_op_in_progress'

    # A writer that dies after its first page: the record exists, half the items are tagged.
    n = 1_200
    (tmp_path / 'big').mkdir()
    big = build_env(
        monkeypatch, tmp_path / 'big', {f'c{i:05d}': proposal(f'c{i:05d}') for i in range(n)}
    )
    big.fake.fail_mget_on_call = 3  # page one reads in two chunks of 256
    resp = big.client.put(
        f'{big.prefix}/crops/batch_label',
        json={'class_id': big.class_ids['widget'], 'crop_ids': _ids(*range(n))},
    )
    assert resp.status_code == 500
    big.fake.fail_mget_on_call = None
    (listed,) = big.list_ops().json()['items']
    assert listed['status'] == 'interrupted'
    assert listed['count'] is None
    assert listed['current'] == OCC_BULK_PAGE_SIZE
    assert listed['undoable'] is True
    assert not (big.state() / listed['op_id'] / 'members.json').exists()

    body = _undo_ok(big, listed['op_id'])
    assert body['members_known'] is False
    assert body['restored'] == OCC_BULK_PAGE_SIZE
    assert all(not d.get('class_validated') for d in big.docs().values())


def test_unknown_malformed_or_pruned_op_id_is_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    env.label('widget', crop_ids=_ids(0))
    # A real record where a joined, unchecked id would land.
    decoy = env.state().parent / 'decoy'
    decoy.mkdir()
    op_id = env.list_ops().json()['items'][0]['op_id']
    (decoy / 'state.json').write_text((env.state() / op_id / 'state.json').read_text())

    opened: list[Path] = []
    real_read = Path.read_text

    def spy(self: Path, *a: Any, **kw: Any) -> str:
        opened.append(self)
        return real_read(self, *a, **kw)

    monkeypatch.setattr(Path, 'read_text', spy)
    for bad in (
        'lop_20260101T000000000000_abcdef',
        'nope',
        '..%2Fdecoy',
        '..',
        'lop_20260101T000000000000_ABCDEF',
    ):
        for dry_run in (True, False):
            resp = env.undo(bad, dry_run=dry_run)
            assert resp.status_code == 404, (bad, resp.text)
    assert (
        env.client.post(f'{env.prefix}/label_ops/%2E%2E%2Fdecoy/undo', json={}).status_code == 404
    )
    root = env.state().resolve()
    outside = [p for p in opened if root not in p.resolve().parents]
    assert not outside, f'read outside the project label_ops dir: {outside}'
    with pytest.raises(Exception, match='no label operation'):
        store.read_op('../decoy')


def test_move_ops_undo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = _env(monkeypatch, tmp_path, n=6)
    for crop_id in _ids(4, 5):
        env.docs()[crop_id].update(cluster_id=10_004)
    # Item 1 is already a human widget; the candidate move clears it, undo brings it back.
    human = env.label('widget', crop_ids=_ids(1))
    before = {k: _state(v) for k, v in env.docs().items()}

    to_class = env.move(env.class_ids['gadget'], crop_ids=_ids(0, 2))
    assert env.docs()[_ids(0)[0]]['class_source'] == 'human_move'
    body = _undo_ok(env, to_class['op_id'])
    assert (body['restored'], body['kept_changed']) == (2, 0)

    to_candidate = env.move(10_004, crop_ids=_ids(1, 3))
    assert env.docs()[_ids(1)[0]]['cluster_id'] == 10_004
    assert env.docs()[_ids(1)[0]]['class_validated'] is False
    # A person pulls item 3 into another candidate cluster afterwards.
    env.docs()[_ids(3)[0]]['cluster_id'] = 10_005
    body = _undo_ok(env, to_candidate['op_id'])
    assert (body['restored'], body['kept_changed']) == (1, 1)
    assert {k: _state(v) for k, v in env.docs().items() if k != _ids(3)[0]} == {
        k: v for k, v in before.items() if k != _ids(3)[0]
    }
    assert human['op_id'] != to_class['op_id']


def test_the_op_record_survives_and_holds_the_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    op = env.label('widget', crop_ids=_ids(0, 1))
    _undo_ok(env, op['op_id'])
    record = json.loads((env.state() / op['op_id'] / 'state.json').read_text())
    assert record['status'] == 'undone'
    assert len(record['undo_passes']) == 1
    assert record['undo_passes'][0]['restored'] == 2
    (listed,) = env.list_ops().json()['items']
    assert listed['status'] == 'undone'
