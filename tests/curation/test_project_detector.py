"""A project's own ingest detector: validated against Triton on write, laid over
the env profile for ingest, the detector info and class seeding, and counted as
a user of the model."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import SettingsFakeOpenSearch
from src.clients.curation_opensearch import ClassRegistry
from src.config import DetectionProfile, get_curation_config
from src.routers.curation import _common
from src.services.curation.ingest_detector import detector_problems, effective_profile
from src.services.curation.ingest_policy import DetectorOverride


URL = '/curation/projects/default/ingest/policy'
END2END = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']


class _Pool:
    def __init__(self, ready: bool = True, outputs: list[str] | None = None) -> None:
        self.ready = ready
        self.outputs = END2END if outputs is None else outputs

    async def is_model_ready(self, _name: str) -> bool:
        return self.ready

    async def get_model_output_names(self, _name: str) -> list[str]:
        return self.outputs


@pytest.mark.asyncio
async def test_a_ready_end2end_model_has_no_problems() -> None:
    assert await detector_problems(_Pool(), DetectorOverride(model='m')) == []


@pytest.mark.asyncio
async def test_a_model_that_is_not_loaded_or_lacks_the_outputs_is_refused() -> None:
    override = DetectorOverride(model='m')
    assert 'not loaded' in (await detector_problems(_Pool(ready=False), override))[0]
    fused = _Pool(outputs=['output0'])
    assert 'end2end outputs' in (await detector_problems(fused, override))[0]


def test_the_override_is_laid_over_the_env_profile_and_never_assigns_by_id() -> None:
    base = DetectionProfile(
        name='item', detector_model='env_model', assigns_class=True, input_size=640
    )
    out = effective_profile(
        base, DetectorOverride(model='mine', version='3', input_size=1280, labels_path='/l.txt')
    )
    assert (out.detector_model, out.detector_version, out.input_size) == ('mine', '3', 1280)
    assert (out.labels_path, out.assigns_class, out.name) == ('/l.txt', False, 'item')
    assert effective_profile(base, None) is base


@pytest.fixture
def labels(tmp_path: Any) -> str:
    path = tmp_path / 'labels.txt'
    path.write_text('bird\nboat\n')
    return str(path)


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    fake = SettingsFakeOpenSearch()
    registry = ClassRegistry(path=tmp_path / 'class_registry.json')
    monkeypatch.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'env_model')
    monkeypatch.setattr(_common, '_INDEXES_BOOTSTRAPPED', {'default'})
    monkeypatch.setattr('src.routers.curation.get_class_registry', lambda: registry)
    monkeypatch.setattr('src.routers.curation.class_seed.get_class_registry', lambda: registry)
    monkeypatch.setattr('src.routers.curation.ingest_policy.get_class_registry', lambda: registry)
    pool = _Pool()
    monkeypatch.setattr('src.main.get_async_triton_pool', lambda: pool)
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    with TestClient(app) as client:
        yield client, fake, pool, registry


def test_put_validates_the_detector_against_triton(world: Any, labels: str) -> None:
    client, fake, pool, _ = world
    body = {'expected_revision': 0, 'detector': {'model': 'birds', 'labels_path': labels}}
    pool.ready = False
    refused = client.put(URL, json=body)
    assert refused.status_code == 422
    assert 'not loaded' in refused.text
    assert fake.write_calls == 0
    pool.ready, pool.outputs = True, ['output0']
    assert client.put(URL, json=body).status_code == 422
    pool.outputs = END2END
    ok = client.put(URL, json=body)
    assert ok.status_code == 200, ok.text
    assert client.get(URL).json()['detector']['model'] == 'birds'


def test_the_project_detector_drives_the_info_and_class_seeding(world: Any, labels: str) -> None:
    client, _, _, registry = world
    assert client.get('/curation/projects/default/ingest/config').json()['detector']['model'] == (
        'env_model'
    )
    client.put(
        URL, json={'expected_revision': 0, 'detector': {'model': 'birds', 'labels_path': labels}}
    )

    detector = client.get('/curation/projects/default/ingest/config').json()['detector']
    assert detector['model'] == 'birds'
    assert detector['assigns_class'] is False
    assert [label['slug'] for label in detector['labels']] == ['bird', 'boat']

    seeded = client.post(
        '/curation/projects/default/classes/seed_from_detector', json={'dry_run': False}
    ).json()
    assert [c['name'] for c in seeded['created']] == ['bird', 'boat']
    assert [c.class_name for c in registry.load().classes] == ['bird', 'boat']


@pytest.mark.asyncio
async def test_a_model_a_project_runs_as_its_ingest_detector_counts_as_in_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.config_store import project_usage
    from src.services.curation.ingest_policy import IngestPolicyBody
    from src.services.curation.ingest_policy_store import put_ingest_policy

    fake = SettingsFakeOpenSearch()
    await put_ingest_policy(
        fake, IngestPolicyBody(detector=DetectorOverride(model='birds')), expected_revision=0
    )

    async def one_project(read: Any) -> dict[str, Any]:
        return {'alpha': await read(get_curation_config().configs_index)}

    monkeypatch.setattr(project_usage, 'read_each_project', one_project)
    assert await project_usage.ingest_detector_users(fake, 'birds') == ['alpha']
    assert await project_usage.ingest_detector_users(fake, 'other') == []
    assert ('alpha', 'ingest detector') in await project_usage.active_detector_users(fake, 'birds')


@pytest.mark.asyncio
async def test_the_ingest_service_runs_the_project_detector_and_its_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.main as main_module
    from src.routers.curation import ingest as ingest_router
    from src.services.curation.ingest_policy import DetectFilter, IngestPolicyBody
    from src.services.curation.ingest_policy_store import put_ingest_policy

    monkeypatch.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'env_model')
    monkeypatch.setattr(main_module, 'get_async_triton_pool', lambda: object())
    monkeypatch.setattr(main_module.app.state, 'pe_encoder', object(), raising=False)
    fake = SettingsFakeOpenSearch()

    plain = await ingest_router._get_ingest_service(fake, object())
    assert plain.profile.detector_model == 'env_model'

    await put_ingest_policy(
        fake,
        IngestPolicyBody(
            detect=DetectFilter(class_resolution='by_name'),
            detector=DetectorOverride(model='birds'),
        ),
        expected_revision=0,
    )
    mine = await ingest_router._get_ingest_service(fake, object())
    assert mine.profile.detector_model == 'birds'
    assert mine.detector.profile is mine.profile
    assert mine.policy.detect.class_resolution == 'by_name'


def test_deleting_the_model_the_stored_ingest_policy_names_is_refused(
    world: Any, labels: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#75: the detector_in_use guard fires for a project-owned end2end model set as
    the ingest detector through the real policy route (not a mocked policy). Fused
    single-output promoted models never pass the servable check, so they never reach
    it; a hand-installed end2end model owned by the project does."""
    from unittest.mock import AsyncMock

    from src.services.training.triton_promote import UnloadResult

    client, _fake, _pool, _registry = world
    unload = AsyncMock(return_value=UnloadResult('wheel_det', True, True))
    monkeypatch.setattr('src.routers.curation.models.unload_triton_model', unload)
    put = client.put(
        URL,
        json={'expected_revision': 0, 'detector': {'model': 'wheel_det', 'labels_path': labels}},
    )
    assert put.status_code == 200, put.text

    refused = client.delete('/curation/projects/default/models/wheel_det')
    assert refused.status_code == 409, refused.text
    assert refused.json()['detail']['error'] == 'detector_in_use'
    unload.assert_not_awaited()

    forced = client.delete('/curation/projects/default/models/wheel_det', params={'force': 'true'})
    assert forced.status_code == 200, forced.text
    unload.assert_awaited_once()
