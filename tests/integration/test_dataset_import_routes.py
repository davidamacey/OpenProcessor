"""``/datasets/*`` read routes through the real app: preview and the format catalog. The
OpenSearch/Triton/PE boundary is faked. Start, poll, resume, undo and upload are queued jobs
and are tested on a database in ``tests/families_b/``."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import pytest
from curation.dataset_import.harness import write_yolo
from curation.query_fakes import SettingsFakeOpenSearch

from integration.ingest_fakes import FakeTritonPool, curation_app


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

BASE = '/curation/projects/default/datasets'
SAME = '0 0.3 0.3 0.4 0.4'


@pytest.fixture
def fake_os() -> SettingsFakeOpenSearch:
    return SettingsFakeOpenSearch()


@pytest.fixture
def client(
    fake_os: SettingsFakeOpenSearch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    from _class_registry_fake import InMemoryRegistry

    registry = InMemoryRegistry()
    registry.add('car')
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


def test_preview_reports_counts_suggestions_and_writes_nothing(
    client: TestClient, fake_os: SettingsFakeOpenSearch, tmp_path: Path
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


def test_formats_serves_the_catalog(client: TestClient) -> None:
    from src.services.curation.dataset_import.issues import ISSUE_CATALOG
    from src.services.curation.dataset_import.options import SourceFormat

    body = client.get(f'{BASE}/formats').json()
    assert {i['code'] for i in body['issues']} == set(ISSUE_CATALOG)
    assert {c['value'] for c in body['mapping_actions']} == {'map', 'create', 'skip', 'region'}
    assert body['upload_limits']['max_bytes'] > 0
    assert {f['format'] for f in body['formats']} == set(get_args(SourceFormat))


@pytest.mark.parametrize('tagged', ['!!bool abc', '!!timestamp abc'])
def test_preview_of_a_data_yaml_with_a_bad_tagged_scalar_is_an_issue_not_a_500(
    client: TestClient, tmp_path: Path, tagged: str
) -> None:
    root = _dataset(tmp_path)
    (root / 'data.yaml').write_text(f'train: images/train\nnames: [{tagged}]\n')
    resp = client.post(f'{BASE}/preview', json={'source': {'path': str(root)}})
    assert resp.status_code == 200, resp.text
    assert 'data_yaml_invalid' in {i['code'] for i in resp.json()['issues']}
