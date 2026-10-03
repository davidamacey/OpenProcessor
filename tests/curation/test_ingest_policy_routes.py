"""`GET/PUT /ingest/policy` and `POST /ingest/policy/preview` over an in-memory index."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import SettingsFakeOpenSearch
from src.clients.curation_opensearch import ClassRegistry
from src.config import IndexRole, get_curation_config, index_name


URL = '/curation/projects/default/ingest/policy'


def _item(i: int, name: str, conf: float, area: float, image: str) -> tuple[str, dict[str, Any]]:
    return f'c{i}', {
        'crop_id': f'c{i}',
        'image_id': image,
        'proposal_name': name,
        'confidence': conf,
        'crop_area_norm': area,
    }


@pytest.fixture
def fake() -> SettingsFakeOpenSearch:
    items = dict(
        [
            _item(1, 'person', 0.9, 0.2, 'i1'),
            _item(2, 'car', 0.8, 0.1, 'i1'),
            _item(3, 'car', 0.4, 0.05, 'i2'),
            _item(4, 'traffic light', 0.7, 0.01, 'i2'),
        ]
    )
    return SettingsFakeOpenSearch({get_curation_config().items_index: items})


@pytest.fixture
def client(fake: SettingsFakeOpenSearch, tmp_path: Any, monkeypatch: pytest.MonkeyPatch):
    registry = ClassRegistry(path=tmp_path / 'class_registry.json')
    monkeypatch.setattr('src.routers.curation.get_class_registry', lambda: registry)
    monkeypatch.setattr('src.routers.curation.ingest_policy.get_class_registry', lambda: registry)

    async def _no_bootstrap(_: Any) -> None:
        return None

    monkeypatch.setattr('src.routers.curation.ingest_policy._ensure_indexes', _no_bootstrap)
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as c:
        yield c


def test_get_returns_defaults_when_never_written(client: TestClient) -> None:
    body = client.get(URL).json()
    assert body['revision'] == 0
    assert body['embedding']['mode'] == 'all'
    assert body['detect'] == {
        'min_confidence': None,
        'min_box_area_frac': None,
        'max_per_image': None,
        'classes': None,
        'exclude_classes': [],
        'class_resolution': 'proposal',
    }
    assert body['detector'] is None


def test_put_round_trips_and_bumps_the_revision(
    client: TestClient, fake: SettingsFakeOpenSearch
) -> None:
    put = {'expected_revision': 0, 'embedding': {'mode': 'selected', 'classes': ['Car', 'unicorn']}}
    first = client.put(URL, json=put)
    assert first.status_code == 200, first.text
    assert first.json()['revision'] == 1
    assert first.json()['unknown_names'] == ['Car', 'unicorn']  # empty registry: both unknown

    got = client.get(URL).json()
    assert got['embedding'] == {
        'min_confidence': None,
        'min_box_area_frac': None,
        'max_per_image': None,
        'mode': 'selected',
        'classes': ['Car', 'unicorn'],
    }
    second = client.put(URL, json={'expected_revision': 1})
    assert second.json()['revision'] == 2
    assert client.get(URL).json()['embedding']['mode'] == 'all'
    settings = fake.docs(index_name(get_curation_config(), IndexRole.SETTINGS))['default']
    assert settings['ingest_policy']['revision'] == 2


def test_stale_expected_revision_is_409_and_writes_nothing(
    client: TestClient, fake: SettingsFakeOpenSearch
) -> None:
    assert client.put(URL, json={'expected_revision': 0}).status_code == 200
    writes = fake.write_calls
    stale = client.put(URL, json={'expected_revision': 0, 'embedding': {'mode': 'lazy'}})
    assert stale.status_code == 409
    assert fake.write_calls == writes
    assert client.get(URL).json()['embedding']['mode'] == 'all'


def test_selected_without_a_criterion_is_422(client: TestClient) -> None:
    r = client.put(URL, json={'expected_revision': 0, 'embedding': {'mode': 'selected'}})
    assert r.status_code == 422


def test_preview_counts_over_stored_items_and_writes_nothing(
    client: TestClient, fake: SettingsFakeOpenSearch
) -> None:
    writes = fake.write_calls
    r = client.post(
        f'{URL}/preview',
        json={'embedding': {'mode': 'selected', 'classes': ['car'], 'min_confidence': 0.5}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body['total_items'], body['scanned'], body['truncated']) == (4, 4, False)
    assert (body['would_embed'], body['would_not_embed']) == (1, 3)  # only c2: car above 0.5
    by = {c['name']: (c['would_embed'], c['would_not_embed']) for c in body['by_class']}
    assert by == {'car': (1, 1), 'person': (0, 1), 'traffic_light': (0, 1)}
    assert body['estimated_vector_mb'] == pytest.approx(1 * 1024 * 4 / 1e6, abs=0.01)
    assert fake.write_calls == writes


def test_preview_caps_per_image_using_one_images_items(client: TestClient) -> None:
    body = client.post(
        f'{URL}/preview', json={'embedding': {'mode': 'selected', 'max_per_image': 1}}
    ).json()
    assert body['would_embed'] == 2  # one per image (i1: person, i2: traffic light 0.7)


class _Pool:
    def __init__(self, *, ready: bool, outputs: list[str]) -> None:
        self._ready, self._outputs = ready, outputs

    async def is_model_ready(self, _model: str) -> bool:
        return self._ready

    async def get_model_output_names(self, _model: str) -> list[str]:
        return self._outputs


def test_an_unservable_detector_override_is_a_typed_422_with_every_reason(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        'src.main.get_async_triton_pool', lambda: _Pool(ready=False, outputs=[]), raising=False
    )
    r = client.put(URL, json={'expected_revision': 0, 'detector': {'model': 'ghost'}})
    assert r.status_code == 422, r.text
    detail = r.json()['detail']
    assert detail['error'] == 'detector_not_servable'
    assert detail['reasons'] == ["'ghost' is not loaded and ready on Triton"]
    assert detail['message'] == detail['reasons'][0]


def test_no_triton_pool_is_a_typed_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_pool() -> Any:
        raise RuntimeError('not started')

    monkeypatch.setattr('src.main.get_async_triton_pool', _no_pool, raising=False)
    r = client.put(URL, json={'expected_revision': 0, 'detector': {'model': 'ghost'}})
    assert r.status_code == 503
    assert r.json()['detail']['error'] == 'detector_unavailable'


def test_a_stale_revision_is_a_typed_409(client: TestClient) -> None:
    r = client.put(URL, json={'expected_revision': 5})
    assert r.status_code == 409
    assert r.json()['detail']['error'] == 'revision_conflict'
