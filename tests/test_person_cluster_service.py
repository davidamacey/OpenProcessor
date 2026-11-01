"""Person clustering/merge/rename/dissolve against an in-memory faces index (#354)."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.services.face_identity import FaceIdentityService


class FakeFaces:
    """Just enough of AsyncOpenSearch for the person writers."""

    def __init__(self, docs: dict[str, dict[str, Any]]):
        self.docs = copy.deepcopy(docs)
        self.seq = dict.fromkeys(self.docs, 1)
        self.bulk_docs: list[tuple[str, dict[str, Any]]] = []
        self.update_by_query_calls: list[dict[str, Any]] = []
        self.hook_before_mget: Any = None
        self.indices = SimpleNamespace(refresh=self._refresh)

    async def _refresh(self, **_: Any) -> dict:
        return {}

    async def search(self, body: dict[str, Any], **_: Any) -> dict:
        ids = sorted(self.docs)
        term = body['query'].get('term')
        if term:
            ((field, value),) = term.items()
            ids = [i for i in ids if self.docs[i].get(field) == value]
        after = body.get('search_after')
        if after:
            ids = [i for i in ids if i > after[0]]
        hits = []
        for i in ids[: body.get('size', 10)]:
            src = dict(self.docs[i])
            spec = body.get('_source')
            if isinstance(spec, dict) and 'embedding' in spec.get('excludes', []):
                src.pop('embedding', None)
            hits.append({'_id': i, '_source': src, 'sort': [i]})
        return {'hits': {'hits': hits}}

    async def mget(self, body: dict[str, Any]) -> dict:
        if self.hook_before_mget:
            self.hook_before_mget(self)
            self.hook_before_mget = None
        out = []
        for d in body['docs']:
            doc = self.docs.get(d['_id'])
            if doc is None:
                out.append({'_id': d['_id'], 'found': False})
                continue
            out.append(
                {
                    '_id': d['_id'],
                    'found': True,
                    '_source': dict(doc),
                    '_seq_no': self.seq[d['_id']],
                    '_primary_term': 1,
                }
            )
        return {'docs': out}

    async def bulk(self, body: list[dict[str, Any]], **_: Any) -> dict:
        items = []
        for action, payload in zip(body[::2], body[1::2], strict=True):
            meta = action['update']
            assert meta['if_seq_no'] == self.seq[meta['_id']]
            self.docs[meta['_id']].update(payload['doc'])
            self.seq[meta['_id']] += 1
            self.bulk_docs.append((meta['_id'], payload['doc']))
            items.append({'update': {'status': 200}})
        return {'items': items}

    async def update(self, body: dict[str, Any], **kw: Any) -> dict:
        self.docs[kw['id']].update(body['doc'])
        return {}

    async def update_by_query(self, body: dict[str, Any], **_: Any) -> dict:
        self.update_by_query_calls.append(body)
        return {'updated': 3}


def _service(fake: FakeFaces) -> FaceIdentityService:
    svc = FaceIdentityService.__new__(FaceIdentityService)
    svc._opensearch = SimpleNamespace(client=fake)  # type: ignore[assignment]
    return svc


def _unit(i: int, d: int = 8) -> list[float]:
    v = np.zeros(d, dtype=np.float32)
    v[i % d] = 1.0
    return v.tolist()


def _doc(fid: str, vec: list[float], **extra: Any) -> dict[str, Any]:
    return {'face_id': fid, 'embedding': vec, **extra}


@pytest.mark.asyncio
async def test_every_face_gets_a_person_including_singletons():
    # a and b are the same direction, c and d are alone
    fake = FakeFaces(
        {
            'a': _doc('a', _unit(0)),
            'b': _doc('b', _unit(0)),
            'c': _doc('c', _unit(1)),
            'd': _doc('d', _unit(2)),
        }
    )
    res = await _service(fake).auto_cluster_faces(0.7, 2)
    assert res['status'] == 'success', res
    ids = {k: v['person_id'] for k, v in fake.docs.items()}
    assert all(ids.values())
    assert ids['a'] == ids['b']
    assert len({ids['a'], ids['c'], ids['d']}) == 3
    assert res['singletons'] == 2
    assert res['persons_total'] == 3

    again = await _service(fake).auto_cluster_faces(0.7, 2)
    assert again['faces_updated'] == 0
    assert again['persons_new'] == 0
    assert {k: v['person_id'] for k, v in fake.docs.items()} == ids


@pytest.mark.asyncio
async def test_no_read_cap_every_face_is_scanned(monkeypatch: pytest.MonkeyPatch):
    n = 100_001
    fake = FakeFaces({f'f{i:06d}': _doc(f'f{i:06d}', [1.0, 0.0]) for i in range(n)})
    import src.services.face_identity.person_management as pm

    monkeypatch.setattr(pm, 'similarity_components', lambda x, _t: [[i] for i in range(len(x))])
    res = await _service(fake).auto_cluster_faces(0.7, 2)
    assert res['status'] == 'success', res
    assert res['total_faces'] == n
    assert all(d.get('person_id') for d in fake.docs.values())


@pytest.mark.asyncio
async def test_locked_face_changed_between_scan_and_write_is_skipped():
    fake = FakeFaces({'a': _doc('a', _unit(0)), 'b': _doc('b', _unit(0))})

    def lock_b(f: FakeFaces) -> None:
        f.docs['b']['named_at'] = '2026-01-01T00:00:00Z'
        f.docs['b']['person_id'] = 'person_human'
        f.seq['b'] += 1

    fake.hook_before_mget = lock_b
    res = await _service(fake).auto_cluster_faces(0.7, 2)
    assert res['faces_skipped'] == 1
    assert fake.docs['b']['person_id'] == 'person_human'


@pytest.mark.asyncio
async def test_naming_locks_and_new_matching_face_joins_the_name():
    fake = FakeFaces(
        {
            'a': _doc('a', _unit(0), person_id='P', person_name='Mom', named_at='t'),
            'b': _doc('b', _unit(0), person_id='P', person_name='Mom', named_at='t'),
            'c': _doc('c', _unit(0)),
        }
    )
    res = await _service(fake).auto_cluster_faces(0.7, 2)
    assert res['status'] == 'success'
    assert fake.docs['c']['person_id'] == 'P'
    assert fake.docs['c']['person_name'] == 'Mom'
    assert {i for i, _ in fake.bulk_docs} == {'c'}
    assert res['faces_locked'] == 2


@pytest.mark.asyncio
async def test_rename_sends_name_as_param_and_stamps_lock():
    fake = FakeFaces({})
    evil = "O'Brien'; ctx._source.x = 1; '"
    res = await _service(fake).rename_person('P', evil)
    assert res['status'] == 'success'
    script = fake.update_by_query_calls[0]['script']
    assert script['params']['name'] == evil
    assert evil not in script['source']
    assert "'" not in script['source']
    assert 'named_at' in script['source']
    assert 'ts' in script['params']


@pytest.mark.asyncio
async def test_merge_uses_params_stamps_both_sides_and_one_name():
    fake = FakeFaces(
        {
            's1': {'face_id': 's1', 'person_id': "S'1", 'person_name': 'Src'},
            't1': {'face_id': 't1', 'person_id': 'T', 'person_name': "T'name"},
        }
    )
    res = await _service(fake).merge_persons("S'1", 'T')
    assert res['status'] == 'success'
    body = fake.update_by_query_calls[0]
    script = body['script']
    assert "'" not in script['source']
    assert "S'1" not in script['source']
    p = script['params']
    assert p['target'] == 'T'
    assert p['source'] == "S'1"
    assert p['name'] == "T'name"
    assert set(body['query']['terms']['person_id']) == {"S'1", 'T'}
    assert 'merged_at' in script['source']
    assert 'merged_from' in script['source']


@pytest.mark.asyncio
async def test_dissolve_clears_lock_stamps():
    fake = FakeFaces(
        {
            'a': {
                'face_id': 'a',
                'person_id': 'P',
                'person_name': 'Mom',
                'named_at': 't',
                'merged_at': 't',
                'merged_from': 'Q',
            }
        }
    )
    res = await _service(fake).delete_person('P', delete_faces=False)
    assert res['status'] == 'success'
    d = fake.docs['a']
    assert d['person_id'] == 'person_a'
    assert d['person_name'] is None
    assert d['named_at'] is None
    assert d['merged_at'] is None
    assert d['merged_from'] is None
