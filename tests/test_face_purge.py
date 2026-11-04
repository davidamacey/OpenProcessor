"""The face-data purge: dry-run counts, full removal, idempotence, and its refusals.

The OpenSearch client is an in-memory fake that records every mutating call.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from scripts import purge_face_data
from src.services.face_privacy import FacePurgeIncompleteError, face_data_counts


if TYPE_CHECKING:
    from pathlib import Path

FACES = 'visual_search_faces'
PEOPLE = 'visual_search_people'
SECRETS = ('person-aaa', 'person-bbb', 'person-ccc', 'Alice Example', 'Bob Example')


def _last_json(out: str) -> Any:
    """The report is the last stdout line; structured log lines may precede it."""
    return json.loads(out.strip().splitlines()[-1])


def _face(i: int, person: str, name: str | None, thumb: bool = False) -> dict[str, Any]:
    doc: dict[str, Any] = {'face_id': f'f{i}', 'person_id': person, 'embedding': [0.1, 0.2]}
    if name:
        doc['person_name'] = name
    if thumb:
        doc['thumbnail_b64'] = 'AAAA'
    return doc


def _matches(doc: dict[str, Any], query: dict[str, Any]) -> bool:
    if 'match_all' in query:
        return True
    if 'exists' in query:
        return doc.get(query['exists']['field']) is not None
    if 'term' in query:
        ((field, value),) = query['term'].items()
        return doc.get(field) == value
    clauses = query['bool']
    if 'must' in clauses:
        return all(_matches(doc, q) for q in clauses['must'])
    return any(_matches(doc, q) for q in clauses['should'])


class FakeIndices:
    def __init__(self, os_: FakeOpenSearch) -> None:
        self.os = os_

    async def exists(self, index: str) -> bool:
        return index in self.os.docs

    async def delete(self, index: str) -> None:
        self.os.mutations.append(('indices.delete', index))
        if not self.os.delete_is_noop:
            self.os.docs.pop(index, None)

    async def refresh(self, index: str) -> None:
        self.os.mutations.append(('indices.refresh', index))


class FakeOpenSearch:
    def __init__(self, delete_is_noop: bool = False) -> None:
        self.delete_is_noop = delete_is_noop
        self.docs: dict[str, list[dict[str, Any]]] = {
            FACES: [
                _face(0, 'person-aaa', 'Alice Example', thumb=True),
                _face(1, 'person-aaa', 'Alice Example'),
                _face(2, 'person-bbb', 'Bob Example'),
                _face(3, 'person-ccc', None),
                _face(4, 'person-ccc', None),
            ],
            PEOPLE: [{'image_id': 'img', 'face_id': 'f0', 'has_face': True}],
        }
        self.mutations: list[tuple[str, Any]] = []
        self.indices = FakeIndices(self)

    def _hits(self, index: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        return [d for d in self.docs[index] if _matches(d, body['query'])]

    async def count(self, index: str, body: dict[str, Any]) -> dict[str, int]:
        return {'count': len(self._hits(index, body))}

    async def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        ids = sorted({d['person_id'] for d in self._hits(index, body) if d.get('person_id')})
        comp = body['aggs']['persons']['composite']
        start = 0
        if 'after' in comp:
            start = ids.index(comp['after']['p']) + 1
        page = ids[start : start + comp['size']]
        agg: dict[str, Any] = {'buckets': [{'key': {'p': i}} for i in page]}
        if page:
            agg['after_key'] = {'p': page[-1]}
        return {'aggregations': {'persons': agg}}

    async def close(self) -> None:
        return None

    async def update_by_query(self, index: str, body: dict[str, Any], **_kw: Any) -> dict[str, int]:
        self.mutations.append(('update_by_query', index))
        hits = self._hits(index, body)
        for d in hits:
            d.pop('face_id', None)
            d['has_face'] = False
        return {'updated': len(hits)}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    index_dir = tmp_path / 'faiss'
    index_dir.mkdir()
    for name in ('faces.index', 'faces.meta.json', 'faces.index.tmp', 'global.index'):
        (index_dir / name).write_bytes(b'x')
    monkeypatch.chdir(tmp_path)
    fake = FakeOpenSearch()
    monkeypatch.setattr(purge_face_data, 'make_client', lambda: fake)
    return {'fake': fake, 'dir': index_dir, 'root': tmp_path}


@pytest.mark.asyncio
async def test_counts_report_exactly_what_is_stored(env: dict[str, Any]) -> None:
    counts = await face_data_counts(env['fake'], env['dir'])
    assert counts.face_index_exists is True
    assert counts.face_docs == 5
    assert counts.persons == 3
    assert counts.named_persons == 2
    assert counts.thumbnails == 1
    assert counts.people_face_links == 1
    assert sorted(counts.faiss_files) == ['faces.index', 'faces.index.tmp', 'faces.meta.json']


def test_dry_run_prints_the_counts_and_changes_nothing(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    rc = purge_face_data.main(['--dry-run', '--index-dir', str(env['dir'])])
    out = _last_json(capsys.readouterr().out)
    assert rc == 0
    assert out['face_docs'] == 5
    assert out['persons'] == 3
    assert env['fake'].mutations == []
    assert (env['dir'] / 'faces.index').exists()
    assert not (env['root'] / 'data' / 'audit').exists()


def test_yes_deletes_everything_and_audits_counts_only(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv('OP_FACES_ENABLED', 'false')
    fake: FakeOpenSearch = env['fake']
    rc = purge_face_data.main(['--yes', '--index-dir', str(env['dir'])])
    assert rc == 0
    assert [m for m in fake.mutations if m[0] == 'indices.delete'] == [('indices.delete', FACES)]
    assert FACES not in fake.docs
    assert 'face_id' not in fake.docs[PEOPLE][0]
    assert fake.docs[PEOPLE][0]['has_face'] is False
    assert len(fake.docs[PEOPLE]) == 1  # the people document itself stays
    left = sorted(p.name for p in env['dir'].iterdir())
    assert left == ['global.index']
    report = _last_json(capsys.readouterr().out)
    assert report['after']['face_docs'] == 0
    assert report['after']['faiss_files'] == []
    audit = (env['root'] / 'data' / 'audit' / 'faces_purge.jsonl').read_text().splitlines()
    assert len(audit) == 1
    for secret in (*SECRETS, str(env['dir']), 'f0'):
        assert secret not in audit[0]
    assert json.loads(audit[0])['face_docs'] == 5


def test_a_second_purge_is_a_clean_no_op_with_its_own_audit_line(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_FACES_ENABLED', 'false')
    args = ['--yes', '--index-dir', str(env['dir'])]
    assert purge_face_data.main(args) == 0
    fake: FakeOpenSearch = env['fake']
    fake.mutations.clear()
    assert purge_face_data.main(args) == 0
    assert fake.mutations == []
    audit = (env['root'] / 'data' / 'audit' / 'faces_purge.jsonl').read_text().splitlines()
    assert len(audit) == 2
    assert json.loads(audit[1])['face_docs'] == 0


def test_an_index_that_survives_the_delete_is_a_failure(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_FACES_ENABLED', 'false')
    monkeypatch.setattr(purge_face_data, 'make_client', lambda: FakeOpenSearch(delete_is_noop=True))
    assert purge_face_data.main(['--yes', '--index-dir', str(env['dir'])]) == 1


@pytest.mark.asyncio
async def test_the_service_raises_when_data_remains(env: dict[str, Any]) -> None:
    from src.services.face_privacy import purge_face_data as purge

    with pytest.raises(FacePurgeIncompleteError):
        await purge(FakeOpenSearch(delete_is_noop=True), env['dir'])


def test_no_flag_reports_and_refuses(
    env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    rc = purge_face_data.main(['--index-dir', str(env['dir'])])
    captured = capsys.readouterr()
    assert rc == 2
    assert _last_json(captured.out)['face_docs'] == 5
    assert '--yes' in captured.err
    assert env['fake'].mutations == []
    assert (env['dir'] / 'faces.index').exists()


def test_yes_is_refused_while_faces_are_on(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv('OP_FACES_ENABLED', raising=False)
    rc = purge_face_data.main(['--yes', '--index-dir', str(env['dir'])])
    assert rc == 2
    assert 'OP_FACES_ENABLED=false' in capsys.readouterr().err
    assert env['fake'].mutations == []
    assert (env['dir'] / 'faces.index').exists()


def test_dry_run_and_yes_are_mutually_exclusive(env: dict[str, Any]) -> None:
    with pytest.raises(SystemExit) as exc:
        purge_face_data.main(['--dry-run', '--yes'])
    assert exc.value.code == 2
