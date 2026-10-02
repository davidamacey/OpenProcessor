"""``/datasets/*`` through the real app: preview, start, poll, idempotency,
errors, undo, archive upload. The OpenSearch/Triton/PE boundary is faked;
the routes, the job and the writers are production code."""

from __future__ import annotations

import io
import time
import zipfile
from typing import TYPE_CHECKING

import pytest
from curation.dataset_import.harness import write_yolo
from curation.query_fakes import QueryFakeOpenSearch

from integration.ingest_fakes import FakeTritonPool, curation_app


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

BASE = '/curation/projects/default/datasets'
SAME = '0 0.3 0.3 0.4 0.4'


@pytest.fixture
def fake_os() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch()


@pytest.fixture
def client(
    fake_os: QueryFakeOpenSearch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    from src.clients.curation_opensearch import ClassRegistry

    monkeypatch.setenv('OP_DATASET_IMPORTS_DIR', str(tmp_path / 'imports'))
    registry = ClassRegistry(path=tmp_path / 'class_registry.json')
    registry.add_class('car')
    with curation_app(fake_os, FakeTritonPool(), monkeypatch, registry=registry) as c:
        yield c


def _dataset(tmp_path: Path, name: str = 'ds') -> Path:
    root = tmp_path / name
    write_yolo(
        root,
        names=['Car', 'truck'],
        images={'a': ['0 0.3 0.3 0.4 0.4', '1 0.7 0.7 0.2 0.2'], 'b': [], 'c': None},
    )
    return root


def _body(root: Path, **extra) -> dict:
    return {
        'source': {'path': str(root)},
        'mapping': [
            {'dataset_class': 'Car', 'action': 'map', 'class_id': 0},
            {'dataset_class': 'truck', 'action': 'create', 'new_class_name': 'truck'},
        ],
        **extra,
    }


def _wait(client: TestClient, import_id: str, *, until: set[str], timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f'{BASE}/imports/{import_id}').json()
        if job['status'] in until:
            return job
        time.sleep(0.05)
    msg = f'import stuck: {job}'
    raise AssertionError(msg)


def test_preview_reports_counts_suggestions_and_writes_nothing(
    client: TestClient, fake_os: QueryFakeOpenSearch, tmp_path: Path
) -> None:
    root = _dataset(tmp_path)
    resp = client.post(f'{BASE}/preview', json={'source': {'path': str(root)}})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['format'] == 'yolo'
    assert body['totals'] == {
        'images': 3,
        'boxes': 2,
        'images_already_indexed': 0,
        'images_to_ingest': 3,
    }
    classes = {c['dataset_class']: c for c in body['classes']}
    assert classes['Car']['suggestion'] == {
        'action': 'map',
        'class_id': 0,
        'class_name': 'car',
        'match': 'case_insensitive',
    }
    assert classes['truck']['suggestion']['action'] == 'create'
    # The id-for-id path would have called dataset class 1 (truck) whatever
    # the registry's class 1 is; here there is none.
    assert classes['Car']['index_would_have_mapped_to'] == {'class_id': 0, 'class_name': 'car'}
    assert {i['code'] for i in body['issues']} >= {'label_file_missing'}
    assert (
        body['import_key']
        == client.post(f'{BASE}/preview', json={'source': {'path': str(root)}}).json()['import_key']
    )
    assert fake_os.bulk_calls == 0
    assert not any(fake_os.store.values())


def test_start_runs_to_completion_and_is_idempotent(
    client: TestClient, fake_os: QueryFakeOpenSearch, tmp_path: Path
) -> None:
    root = _dataset(tmp_path)
    started = client.post(f'{BASE}/imports', json=_body(root))
    assert started.status_code == 202, started.text
    import_id = started.json()['import_id']
    job = _wait(client, import_id, until={'completed', 'completed_with_errors', 'failed'})
    assert job['status'] == 'completed', job
    assert job['report']['items_created'] == 2
    assert job['report']['negatives'] == 1
    assert job['report']['unlabeled'] == 1
    assert {m['dataset_class']: m['class_name'] for m in job['mapping']} == {
        'Car': 'car',
        'truck': 'truck',
    }
    writes = fake_os.bulk_calls
    again = client.post(f'{BASE}/imports', json=_body(root))
    assert again.status_code == 200, again.text
    assert again.json()['reused'] is True
    assert again.json()['import_id'] == import_id
    assert fake_os.bulk_calls == writes
    entries = client.get(f'{BASE}/imports/{import_id}/entries?page_size=10').json()
    assert entries['total'] == 3
    assert {e['status'] for e in entries['items']} == {'ok'}


@pytest.mark.parametrize(
    ('mutate', 'status', 'code'),
    [
        (lambda b: b.update(mapping=[]), 422, 'class_mapping_incomplete'),
        (
            lambda b: b['mapping'].append({'dataset_class': 'nope', 'action': 'skip'}),
            422,
            'class_mapping_invalid',
        ),
        (
            lambda b: b['mapping'].__setitem__(
                1, {'dataset_class': 'truck', 'action': 'create', 'new_class_name': 'CAR'}
            ),
            422,
            'class_mapping_invalid',
        ),
        (lambda b: b.update(expected_import_key='0' * 64), 409, 'dataset_changed'),
        (lambda b: b['source'].update(path='/etc'), 422, 'dataset_path_not_allowed'),
        (lambda b: b['source'].update(path='/nonexistent/x'), 422, 'dataset_path_not_allowed'),
        (lambda b: b.update(surprise=True), 422, None),
    ],
)
def test_start_rejects_every_bad_request(
    client: TestClient, tmp_path: Path, mutate, status: int, code: str | None
) -> None:
    body = _body(_dataset(tmp_path))
    mutate(body)
    resp = client.post(f'{BASE}/imports', json=body)
    assert resp.status_code == status, resp.text
    if code is not None:
        assert resp.json()['detail']['error'] == code


def test_blocking_issue_refuses_the_import(client: TestClient, tmp_path: Path) -> None:
    root = tmp_path / 'broken'
    root.mkdir()
    (root / 'data.yaml').write_text('train: images/train\nnames: [unterminated\n')
    resp = client.post(f'{BASE}/imports', json={'source': {'path': str(root)}})
    assert resp.status_code == 422, resp.text
    detail = resp.json()['detail']
    assert detail['error'] == 'import_blocked'
    assert 'data_yaml_invalid' in {i['code'] for i in detail['issues']}


def test_import_ids_are_never_paths(client: TestClient) -> None:
    for bad in ('..', '%2e%2e', 'imp_x', 'imp_20260101T000000_zzzzzzzz'):
        assert client.get(f'{BASE}/imports/{bad}').status_code in {404}
        assert client.post(f'{BASE}/imports/{bad}/cancel').status_code == 404
        assert client.post(f'{BASE}/imports/{bad}/undo', json={'dry_run': True}).status_code == 404
    assert client.get(f'{BASE}/imports/nope').json()['detail']['error'] == 'import_not_found'


def test_undo_dry_run_then_apply(
    client: TestClient, fake_os: QueryFakeOpenSearch, tmp_path: Path
) -> None:
    root = _dataset(tmp_path)
    import_id = client.post(f'{BASE}/imports', json=_body(root)).json()['import_id']
    _wait(client, import_id, until={'completed'})
    assert sum(len(docs) for docs in fake_os.store.values()) > 0  # the import wrote something
    dry = client.post(f'{BASE}/imports/{import_id}/undo', json={'dry_run': True})
    assert dry.status_code == 200, dry.text
    assert dry.json()['items_deleted'] == 2
    assert dry.json()['classes_deprecated'] == ['truck']
    applied = client.post(f'{BASE}/imports/{import_id}/undo', json={'dry_run': False})
    assert applied.status_code == 202, applied.text
    job = _wait(client, import_id, until={'undone'})
    assert job['undo']['items_deleted'] == 2
    again = client.post(f'{BASE}/imports/{import_id}/undo', json={'dry_run': True}).json()
    assert again['items_deleted'] == 0
    # An undone key can be imported again.
    second = client.post(f'{BASE}/imports', json=_body(root))
    assert second.status_code == 202, second.text


def test_undo_is_refused_while_another_import_is_live(client: TestClient, tmp_path: Path) -> None:
    from src.services.curation.file_job import FileJob

    root = _dataset(tmp_path)
    import_id = client.post(f'{BASE}/imports', json=_body(root)).json()['import_id']
    _wait(client, import_id, until={'completed'})
    other = FileJob(tmp_path / 'imports' / 'projects' / 'default' / 'imp_20990101T000000_deadbeef')
    other.write({'status': 'running'})
    other.touch_heartbeat()
    refused = client.post(f'{BASE}/imports/{import_id}/undo', json={'dry_run': False})
    assert refused.status_code == 409, refused.text
    assert refused.json()['detail']['error'] == 'import_busy'
    assert client.get(f'{BASE}/imports/{import_id}').json()['status'] == 'completed'


def _zip_bytes(root: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                zf.write(path, path.relative_to(root.parent).as_posix())
    return buf.getvalue()


def test_upload_then_preview_matches_the_directory_preview(
    client: TestClient, tmp_path: Path
) -> None:
    root = _dataset(tmp_path)
    direct = client.post(f'{BASE}/preview', json={'source': {'path': str(root)}}).json()
    up = client.post(
        f'{BASE}/uploads', files={'file': ('ds.zip', _zip_bytes(root), 'application/zip')}
    )
    assert up.status_code == 201, up.text
    uploaded = client.post(
        f'{BASE}/preview', json={'source': {'path': up.json()['dataset_path']}}
    ).json()
    assert uploaded['totals'] == direct['totals']
    assert [c['dataset_class'] for c in uploaded['classes']] == [
        c['dataset_class'] for c in direct['classes']
    ]


def test_upload_rejections(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    traversal = io.BytesIO()
    with zipfile.ZipFile(traversal, 'w') as zf:
        zf.writestr('../escape.txt', 'x')
    resp = client.post(f'{BASE}/uploads', files={'file': ('a.zip', traversal.getvalue())})
    assert resp.status_code == 422
    assert resp.json()['detail']['error'] == 'archive_invalid'
    monkeypatch.setenv('OP_DATASET_UPLOAD_MAX_BYTES', '100')
    big = client.post(f'{BASE}/uploads', files={'file': ('a.zip', b'x' * 5000)})
    assert big.status_code == 413
    assert big.json()['detail']['error'] == 'upload_too_large'
    assert big.json()['detail']['limit'] == 100


def test_formats_serves_the_catalog(client: TestClient) -> None:
    from src.services.curation.dataset_import.issues import ISSUE_CATALOG

    body = client.get(f'{BASE}/formats').json()
    assert {i['code'] for i in body['issues']} == set(ISSUE_CATALOG)
    assert {c['value'] for c in body['mapping_actions']} == {'map', 'create', 'skip', 'region'}
    assert body['upload_limits']['max_bytes'] > 0


def test_upload_accepts_a_zip_without_file_type_bits(client: TestClient, tmp_path: Path) -> None:
    """``ZipFile.writestr`` stores permission bits only; such an archive is
    ordinary, not "non-regular"."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('ds/data.yaml', 'train: images/train\nnames: [car]\n')
    resp = client.post(f'{BASE}/uploads', files={'file': ('a.zip', buf.getvalue())})
    assert resp.status_code == 201, resp.text


@pytest.mark.parametrize('tagged', ['!!bool abc', '!!timestamp abc'])
def test_preview_of_a_data_yaml_with_a_bad_tagged_scalar_is_an_issue_not_a_500(
    client: TestClient, tmp_path: Path, tagged: str
) -> None:
    root = _dataset(tmp_path)
    (root / 'data.yaml').write_text(f'train: images/train\nnames: [{tagged}]\n')
    resp = client.post(f'{BASE}/preview', json={'source': {'path': str(root)}})
    assert resp.status_code == 200, resp.text
    assert 'data_yaml_invalid' in {i['code'] for i in resp.json()['issues']}


def test_a_bad_mapping_row_says_what_is_wrong(client: TestClient, tmp_path: Path) -> None:
    body = _body(_dataset(tmp_path))
    body['mapping'].append({'dataset_class': 'nope', 'action': 'skip'})
    detail = client.post(f'{BASE}/imports', json=body).json()['detail']
    assert detail['error'] == 'class_mapping_invalid'
    assert (
        'nope: class_mapping_invalid (this dataset has no class with that name' in detail['message']
    )
