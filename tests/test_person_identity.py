"""Stable person ids: pure assignment rules (#354)."""

from __future__ import annotations

import itertools
import random

from src.domain.person_identity import (
    FaceState,
    PersonAssignment,
    assign_person_ids,
    is_person_locked,
)


def _minter():
    counter = itertools.count(1)
    return lambda: f'person_new_{next(counter):04d}'


def _face(fid, pid=None, name=None, locked=False):
    return FaceState(face_id=fid, person_id=pid, person_name=name, locked=locked)


def _apply(faces, assignment: PersonAssignment):
    by_id = {w.face_id: w for w in assignment.writes}
    out = []
    for f in faces:
        w = by_id.get(f.face_id)
        if w is None:
            out.append(f)
        else:
            out.append(_face(f.face_id, w.person_id, w.person_name, f.locked))
    return out


def test_is_person_locked_reads_named_at_and_merged_at():
    assert not is_person_locked({})
    assert not is_person_locked({'named_at': None, 'merged_at': ''})
    assert is_person_locked({'named_at': '2026-01-01T00:00:00Z'})
    assert is_person_locked({'merged_at': '2026-01-01T00:00:00Z'})


def test_singleton_gets_unique_id():
    faces = [_face('a'), _face('b'), _face('c')]
    res = assign_person_ids(faces, [[0], [1], [2]], min_faces=2, mint=_minter())
    ids = {w.person_id for w in res.writes}
    assert len(res.writes) == 3
    assert len(ids) == 3
    assert res.singletons == 3

    pair = assign_person_ids(faces[:2], [[0, 1]], min_faces=3, mint=_minter())
    assert len({w.person_id for w in pair.writes}) == 2


def test_rerun_same_data_keeps_every_id():
    faces = [_face(c) for c in 'abcde']
    comps = [[0, 1, 2], [3, 4]]
    first = assign_person_ids(faces, comps, min_faces=2, mint=_minter())
    faces = _apply(faces, first)
    second = assign_person_ids(faces, comps, min_faces=2, mint=_minter())
    assert second.writes == []
    assert second.persons_new == 0
    assert second.persons_kept == 2


def test_new_faces_keep_old_ids_and_only_new_groups_get_new_ids():
    faces = [
        _face('a1', 'PA'),
        _face('a2', 'PA'),
        _face('a3', 'PA'),
        _face('b1', 'PB'),
        _face('b2', 'PB'),
        _face('a4'),
        _face('c1'),
        _face('c2'),
    ]
    comps = [[0, 1, 2, 5], [3, 4], [6, 7]]
    res = assign_person_ids(faces, comps, min_faces=2, mint=_minter())
    after = {f.face_id: f.person_id for f in _apply(faces, res)}
    assert after['a4'] == 'PA'
    assert after['a1'] == 'PA'
    assert after['b1'] == after['b2'] == 'PB'
    assert after['c1'] == after['c2'] == 'person_new_0001'
    assert res.persons_new == 1
    assert {w.face_id for w in res.writes} == {'a4', 'c1', 'c2'}


def test_split_keeps_id_on_larger_part():
    faces = [_face(f'f{i}', 'P') for i in range(5)]
    res = assign_person_ids(faces, [[0, 1, 2], [3, 4]], min_faces=2, mint=_minter())
    after = {f.face_id: f.person_id for f in _apply(faces, res)}
    assert {after['f0'], after['f1'], after['f2']} == {'P'}
    assert after['f3'] == after['f4'] != 'P'


def test_merge_keeps_id_with_most_members():
    faces = [
        _face('p1', 'P'),
        _face('p2', 'P'),
        _face('p3', 'P'),
        _face('q1', 'Q'),
        _face('q2', 'Q'),
    ]
    res = assign_person_ids(faces, [[0, 1, 2, 3, 4]], min_faces=2, mint=_minter())
    assert {f.person_id for f in _apply(faces, res)} == {'P'}
    assert res.persons_retired == 1
    assert res.persons_total == 1


def test_tie_breaks_are_deterministic():
    faces = [_face('a', 'P2'), _face('b', 'P2'), _face('c', 'P1'), _face('d', 'P1')]
    res = assign_person_ids(faces, [[0, 1, 2, 3]], min_faces=2, mint=_minter())
    assert {f.person_id for f in _apply(faces, res)} == {'P1'}

    split = [_face(c, 'P') for c in 'abcd']
    outs = set()
    for seed in range(6):
        order = list(range(4))
        random.Random(seed).shuffle(order)
        shuffled = [split[i] for i in order]
        pos = {f.face_id: i for i, f in enumerate(shuffled)}
        comps = [[pos['a'], pos['b']], [pos['c'], pos['d']]]
        r = assign_person_ids(shuffled, comps, min_faces=2, mint=_minter())
        outs.add(tuple(sorted((f.face_id, f.person_id) for f in _apply(shuffled, r))))
    assert len(outs) == 1
    (only,) = outs
    assert dict(only)['a'] == 'P'
    assert dict(only)['c'] != 'P'


def test_named_and_merged_faces_survive():
    faces = [
        _face('m1', 'MOM', 'Mom', locked=True),
        _face('m2', 'MOM', 'Mom', locked=True),
        _face('m3', 'MOM', 'Mom', locked=True),
        _face('x1', 'X'),
        _face('x2', 'X'),
        _face('j1'),
        _face('l1', 'MOM'),
    ]
    # m1,m2 sit with x1,x2; m3 sits with j1; l1 is alone
    comps = [[0, 1, 3, 4], [2, 5], [6]]
    res = assign_person_ids(faces, comps, min_faces=2, mint=_minter())
    written = {w.face_id: w for w in res.writes}
    assert not ({'m1', 'm2', 'm3'} & set(written))
    assert res.faces_locked == 3
    after = {f.face_id: f for f in _apply(faces, res)}
    for k in ('m1', 'm2', 'm3'):
        assert after[k].person_id == 'MOM'
        assert after[k].person_name == 'Mom'
    # an unlocked face leaving Mom's id drops the name; a joiner gains it
    assert after['l1'].person_name is None
    assert after['l1'].person_id != 'MOM'
    assert after['x1'].person_id == 'MOM'
    assert after['x1'].person_name == 'Mom'
