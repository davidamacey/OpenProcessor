"""A detector ``labels_path`` must name a file inside the model repository."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import SettingsFakeOpenSearch
from src.clients.curation_opensearch.registry import ClassRegistry
from src.config.ingest_profiles import load_label_names
from src.routers.curation import _raw_opensearch_dep, router as curation_router
from src.services.curation import index_bootstrap
from src.services.curation.ingest_policy import DetectorOverride


class _Pool:
    async def is_model_ready(self, _name: str) -> bool:
        return True

    async def get_model_output_names(self, _name: str) -> list[str]:
        return ['num_dets', 'det_boxes', 'det_scores', 'det_classes']


if TYPE_CHECKING:
    from pathlib import Path


POLICY_URL = '/curation/projects/default/ingest/policy'


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / 'models'
    (root / 'det').mkdir(parents=True)
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(root))
    load_label_names.cache_clear()
    return root


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    secret = tmp_path / 'secret.txt'
    secret.write_text('TOP-SECRET\n')
    return secret


def _bad_paths(repo: Path, outside: Path) -> list[str]:
    link = repo / 'det' / 'labels.txt'
    link.symlink_to(outside)
    (repo / 'det' / 'notes.json').write_text('{}')
    return [
        '/etc/passwd',
        str(outside),
        str(repo / 'det' / '..' / '..' / 'secret.txt'),
        '../../x.txt',
        str(link),
        str(repo / 'det' / 'notes.json'),
        str(repo / 'det'),
        str(repo / 'det' / 'missing.txt'),
    ]


def test_load_label_names_refuses_a_path_outside_the_repository(repo: Path, outside: Path) -> None:
    for bad in _bad_paths(repo, outside):
        with pytest.raises(ValueError, match='labels_path'):
            load_label_names(bad)


def test_load_label_names_reads_a_labels_file_inside_the_repository(repo: Path) -> None:
    (repo / 'det' / 'labels.txt').write_text('a\n b \n')
    assert load_label_names(str(repo / 'det' / 'labels.txt')) == ('a', 'b')


def test_the_override_rejects_bad_paths_and_accepts_a_good_one(repo: Path, outside: Path) -> None:
    for bad in _bad_paths(repo, outside):
        with pytest.raises(ValueError, match='labels_path'):
            DetectorOverride(model='m', labels_path=bad)
    good = repo / 'det2.txt'
    good.write_text('a\n')
    assert DetectorOverride(model='m', labels_path=str(good)).labels_path == str(good)
    assert DetectorOverride(model='m').labels_path == ''


@pytest.fixture
def policy_client(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    fake = SettingsFakeOpenSearch()
    monkeypatch.setattr(index_bootstrap, '_INDEXES_BOOTSTRAPPED', {'default'})
    registry = ClassRegistry(path=repo.parent / 'class_registry.json')
    monkeypatch.setattr('src.routers.curation.get_class_registry', lambda: registry)
    monkeypatch.setattr('src.routers.curation.class_seed.get_class_registry', lambda: registry)
    monkeypatch.setattr('src.routers.curation.ingest_policy.get_class_registry', lambda: registry)
    monkeypatch.setattr('src.app.resources.get_async_triton_pool', _Pool)
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as client:
        yield client


def test_put_policy_answers_422_for_an_outside_labels_path(
    policy_client: Any, repo: Path, outside: Path
) -> None:
    for bad in _bad_paths(repo, outside):
        body = {'expected_revision': 0, 'detector': {'model': 'm', 'labels_path': bad}}
        resp = policy_client.put(POLICY_URL, json=body)
        assert resp.status_code == 422, bad
        assert 'TOP-SECRET' not in resp.text


@pytest.fixture
def profile_client(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from unittest.mock import AsyncMock

    from curation._fake_config_opensearch import FakeConfigOpenSearch

    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes', AsyncMock(return_value=None)
    )
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = FakeConfigOpenSearch
    return TestClient(app)


def test_region_profile_body_rejects_an_outside_labels_path(
    profile_client: Any, repo: Path, outside: Path
) -> None:
    from curation.test_region_profiles_router import _body

    prefix = '/curation/projects/default/region_profiles'
    for bad in _bad_paths(repo, outside):
        resp = profile_client.post(prefix, json={'name': 'p1', 'body': _body(labels_path=bad)})
        assert resp.status_code == 422, bad
        resp = profile_client.post(prefix + '/validate', json={'body': _body(labels_path=bad)})
        assert resp.status_code == 422, bad
