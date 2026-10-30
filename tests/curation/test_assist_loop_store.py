"""The loop's job directory: an append-only event log that survives a crash mid-line, atomic
JSON files, node member files, and the replay that rebuilds the tree on resume.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from src.services.curation.assist.proposal import ClassAnswer
from src.services.curation.assist_loop.store import LoopStore, replay


if TYPE_CHECKING:
    from pathlib import Path


def _store(tmp_path: Path) -> LoopStore:
    return LoopStore(tmp_path / 'alp_20261028T000000Z_abcdef')


def _lines(store: LoopStore) -> list[str]:
    return store.events_file.read_text(encoding='utf-8').splitlines()


def test_each_event_is_one_json_line_with_an_increasing_sequence(tmp_path: Path) -> None:
    store = _store(tmp_path)

    store.append('node_open', path='r10001', parent=None, n=40, digest='d', level=0)
    store.append('decision', path='r10001', kind='review_leaf')

    events = store.events()
    assert [e['event'] for e in events] == ['node_open', 'decision']
    assert [e['seq'] for e in events] == [1, 2]
    assert all('at' in e for e in events)
    assert len(_lines(store)) == 2
    assert store.loop_id == 'alp_20261028T000000Z_abcdef'


def test_a_reopened_store_continues_the_sequence(tmp_path: Path) -> None:
    _store(tmp_path).append('node_open', path='r1', parent=None, n=1, digest='d', level=0)

    again = _store(tmp_path)
    again.append('decision', path='r1', kind='review_small')

    assert [e['seq'] for e in again.events()] == [1, 2]


def test_a_line_torn_by_a_crash_is_ignored_and_cut_before_the_next_append(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.append('node_open', path='r1', parent=None, n=1, digest='d', level=0)
    with store.events_file.open('a', encoding='utf-8') as fh:
        fh.write('{"seq": 2, "event": "reser')  # the process died mid-write

    reopened = _store(tmp_path)
    assert [e['event'] for e in reopened.events()] == ['node_open']
    reopened.append('decision', path='r1', kind='review_small')

    assert [e['seq'] for e in reopened.events()] == [1, 2]
    assert all(json.loads(line) for line in _lines(reopened))


def test_a_corrupt_line_before_the_end_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.events_file.parent.mkdir(parents=True, exist_ok=True)
    store.events_file.write_text(
        '{"seq": 1, "event": "node_open"}\nnot json\n{"seq": 3, "event": "decision"}\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='events'):
        store.events()


def test_json_files_are_written_whole(tmp_path: Path) -> None:
    store = _store(tmp_path)

    store.write_json('budget.json', {'crops_spent': 3})
    store.write_json('budget.json', {'crops_spent': 4})

    assert store.read_json('budget.json') == {'crops_spent': 4}
    assert sorted(p.name for p in store.directory.iterdir()) == ['budget.json']


def test_a_missing_json_file_reads_as_none(tmp_path: Path) -> None:
    assert _store(tmp_path).read_json('request.json') is None


def test_node_member_files_round_trip_and_a_missing_one_is_an_error(tmp_path: Path) -> None:
    store = _store(tmp_path)

    rel = store.write_node_members('r10001.2', ['c3', 'c1', 'c2'])

    assert rel == 'nodes/r10001.2.json'
    assert store.read_node_members('r10001.2') == ['c1', 'c2', 'c3']
    with pytest.raises(ValueError, match=r'r10001\.3'):
        store.read_node_members('r10001.3')
    with pytest.raises(ValueError, match='not a node path'):
        store.write_node_members('../escape', ['c1'])


def test_roots_are_written_once_and_read_back(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lines = [{'root': 'r10001', 'cluster_id': 10001, 'n': 5, 'digest': 'd', 'members_file': 'x'}]

    store.write_roots(lines)

    assert store.read_roots() == lines
    assert _store(tmp_path).read_roots() == lines
    assert LoopStore(tmp_path / 'other').read_roots() is None


def test_signals_are_files(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert not store.cancel_requested()
    assert not store.pause_requested()

    store.job.request_cancel()
    store.pause_file.parent.mkdir(parents=True, exist_ok=True)
    store.pause_file.touch()

    assert store.cancel_requested()
    assert store.pause_requested()


def _answers(**by_crop: str | None) -> dict[str, Any]:
    return {cid: {'class_name': name, 'proposed': ''} for cid, name in by_crop.items()}


def test_replay_rebuilds_nodes_answers_and_decisions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append('node_open', path='r10001', parent=None, n=60, digest='d0', level=0)
    store.append('reserve', pool='vlm', rid=1, path='r10001', ids=['a', 'b'], crops=2, requests=1)
    store.append(
        'answers', path='r10001', rid=1, answers=_answers(a='car', b='truck'), failed=['x']
    )
    store.append('reserve', pool='vlm', rid=2, path='r10001', ids=['c', 'd'], crops=2, requests=1)
    store.append('decision', path='r10001', kind='split', class_name='car', agree=1)
    store.append(
        'children',
        path='r10001',
        children=[{'path': 'r10001.0', 'n': 30, 'file': 'nodes/r10001.0.json'}],
    )
    store.append('node_open', path='r10001.0', parent='r10001', n=30, digest='d1', level=1)
    store.append('decision', path='r10001.0', kind='pass', class_name='car')
    store.append('propagate_begin', path='r10001.0', class_name='car', n_eligible=20)

    tree = replay(store.events())

    root = tree.nodes['r10001']
    assert root.answers == {'a': ClassAnswer('car'), 'b': ClassAnswer('truck')}
    assert root.failed == {'x'}
    assert root.pending == {2: ['c', 'd']}
    assert root.decision is not None
    assert root.decision['kind'] == 'split'
    assert root.children == ['r10001.0']
    child = tree.nodes['r10001.0']
    assert (child.parent, child.level, child.n) == ('r10001', 1, 30)
    assert child.propagate_begun
    assert child.propagated is None


def test_the_last_decision_of_a_node_wins(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append('node_open', path='r1', parent=None, n=60, digest='d', level=0)
    store.append('decision', path='r1', kind='pass', class_name='car')
    store.append('decision', path='r1', kind='stale', class_name='car')

    assert replay(store.events()).nodes['r1'].decision['kind'] == 'stale'  # type: ignore[index]


def test_a_recovered_answer_settles_its_reservation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append('node_open', path='r1', parent=None, n=60, digest='d', level=0)
    store.append('reserve', pool='vlm', rid=1, path='r1', ids=['a', 'b'], crops=2, requests=1)
    store.append('answers', path='r1', rid=1, answers=_answers(a='car'), failed=[], recovered=True)

    node = replay(store.events()).nodes['r1']

    assert node.answers == {'a': ClassAnswer('car')}
    assert node.pending == {}
    assert 'b' not in node.failed, 'an id with no trace on the item is asked again, not a failure'


def test_audit_reservations_do_not_touch_node_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append('node_open', path='r1', parent=None, n=60, digest='d', level=0)
    store.append('reserve', pool='audit', rid=1, path='r1', ids=['a'], crops=1, requests=1)

    assert replay(store.events()).nodes['r1'].pending == {}
