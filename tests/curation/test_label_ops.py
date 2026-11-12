"""A single label write (``PUT /crops/{id}/label``) is recorded as a label operation: the same
record, tagging and undo as a bulk label, written in the request with no job behind it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from _class_registry_fake import use_registry

from curation.label_ops_env import build_env, proposal
from curation.query_fakes import QueryFakeOpenSearch
from curation.test_crops_undo_exclude import ITEMS, _proposal_doc
from src.domain.label_ops import LABEL_OP_ID_RE


if TYPE_CHECKING:
    from pathlib import Path

    from _class_registry_fake import InMemoryRegistry

    from curation.label_ops_env import OpsEnv

pytest_plugins = ['_records_world', 'curation.jobs_a.optin_queue']

BASE = '/curation/projects/default'


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, n: int = 3) -> OpsEnv:
    return build_env(monkeypatch, tmp_path, {f'c{i}': proposal(f'c{i}') for i in range(n)})


def _put(env: OpsEnv, crop_id: str, class_name: str) -> Any:
    return env.client.put(
        f'{env.prefix}/crops/{crop_id}/label', json={'class_id': env.class_ids[class_name]}
    )


@pytest.mark.db
@pytest.mark.usefixtures('world', 'fake_queue')
def test_single_label_write_is_a_label_op(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = _env(monkeypatch, tmp_path)
    resp = _put(env, 'c0', 'widget')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    op_id = body['op_id']
    assert LABEL_OP_ID_RE.fullmatch(op_id)
    assert (body['crop_id'], body['class_name']) == ('c0', 'widget')

    doc = env.docs()['c0']
    assert doc['label_op_id'] == op_id
    assert doc['class_id_history'][-1]['op_id'] == op_id
    assert 'label_op_id' not in env.docs()['c1']

    record = env.record(op_id)
    assert record['status'] == 'written'
    assert (record['kind'], record['writer'], record['target']) == (
        'label',
        'human:label_crop',
        'crop_ids',
    )
    assert (record['selected'], record['count'], record['conflicts']) == (1, 1, 0)

    served = env.list_ops().json()['items']
    assert [(o['op_id'], o['class_name'], o['current'], o['undoable']) for o in served] == [
        (op_id, 'widget', 1, True)
    ]


@pytest.mark.db
@pytest.mark.usefixtures('world', 'fake_queue')
def test_undo_of_a_single_write_restores_the_previous_class(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    first = _put(env, 'c0', 'widget').json()['op_id']
    second = _put(env, 'c0', 'gadget').json()['op_id']
    assert env.docs()['c0']['class_name'] == 'gadget'

    undone = env.undo(second)
    assert undone.status_code == 200, undone.text
    assert undone.json()['restored'] == 1
    restored = env.docs()['c0']
    assert (restored['class_name'], restored['class_validated']) == ('widget', True)
    assert restored['class_source'] == 'human'

    assert env.undo(first).json()['restored'] == 1
    back = env.docs()['c0']
    assert back['class_validated'] is False
    assert back['class_source'] == 'detector_proposal', 'the machine proposal comes back'


@pytest.mark.db
@pytest.mark.usefixtures('world', 'fake_queue')
def test_undo_of_a_single_write_obeys_the_lock_rule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    changed = _put(env, 'c0', 'widget').json()['op_id']
    _put(env, 'c0', 'gizmo')  # a later person's write owns the item now
    held = _put(env, 'c1', 'widget').json()['op_id']
    env.docs()['c1']['test_holdout'] = True  # frozen after the write

    body = env.undo(changed).json()
    assert (body['restored'], body['kept_changed']) == (0, 1)
    assert env.docs()['c0']['class_name'] == 'gizmo'

    body = env.undo(held).json()
    assert (body['restored'], body['kept_holdout']) == (0, 1)
    assert env.docs()['c1']['class_name'] == 'widget', 'the holdout is never touched'


@pytest.mark.db
@pytest.mark.usefixtures('world', 'fake_queue')
def test_an_unknown_crop_is_404_and_records_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    resp = _put(env, 'nope', 'widget')
    assert resp.status_code == 404, resp.text
    assert env.row_count() == 0


@pytest.mark.db
@pytest.mark.usefixtures('world', 'fake_queue')
def test_a_conflict_that_outlasts_the_retries_is_409(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _env(monkeypatch, tmp_path)
    env.fake.conflict_ids.add('c0')
    resp = _put(env, 'c0', 'widget')
    assert resp.status_code == 409, resp.text
    assert 'label_op_id' not in env.docs()['c0']


def test_the_route_records_one_op_for_one_crop(
    client_for: Any,
    registry: InMemoryRegistry,
    ids: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No store: the route hands one id to the recorded relabel with the single-write
    record (kind, writer, one selected id, no owning job) and answers its ``op_id``."""
    from src.routers.curation import crops_write

    use_registry(registry)
    calls: list[dict[str, Any]] = []

    async def recorded(opensearch: Any, crop_ids: list[str], **kw: Any) -> dict[str, Any]:
        calls.append({'ids': crop_ids, **kw})
        return {'updated': 1, 'updated_ids': crop_ids, 'conflicts': [], 'op_id': 'op-fixed'}

    monkeypatch.setattr(crops_write, 'recorded_relabel', recorded, raising=False)
    fake = QueryFakeOpenSearch({ITEMS: {'c1': _proposal_doc('c1')}})
    resp = client_for(fake).put(f'{BASE}/crops/c1/label', json={'class_id': ids['widget']})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        'crop_id': 'c1',
        'class_id': ids['widget'],
        'class_name': 'widget',
        'op_id': 'op-fixed',
    }
    (call,) = calls
    assert call['ids'] == ['c1']
    assert call['job_id'] is None
    op = call['op']
    assert (op.kind, op.writer, op.class_name, op.target, op.selected) == (
        'label',
        'human:label_crop',
        'widget',
        'crop_ids',
        1,
    )
    tagged = call['merger_for']('op-fixed')(_proposal_doc('c1'))
    assert tagged['label_op_id'] == 'op-fixed'
    assert tagged['class_name'] == 'widget'


def test_an_unknown_crop_is_404_before_any_record(
    client_for: Any,
    registry: InMemoryRegistry,
    ids: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.routers.curation import crops_write

    use_registry(registry)

    async def forbidden(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise AssertionError('no operation is recorded for a missing crop')

    monkeypatch.setattr(crops_write, 'recorded_relabel', forbidden, raising=False)
    fake = QueryFakeOpenSearch({ITEMS: {}})
    resp = client_for(fake).put(f'{BASE}/crops/none/label', json={'class_id': ids['widget']})
    assert resp.status_code == 404, resp.text
