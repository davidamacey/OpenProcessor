"""Reviewer probes (not part of the branch). Each asserts the CORRECT
behavior, so a failure here demonstrates a defect."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._fake_config_opensearch import FakeConfigOpenSearch
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


@pytest.fixture(autouse=True)
def _reset_caches():
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()
    yield
    reset_config_stores()


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    fake_os = FakeConfigOpenSearch()
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    client = TestClient(app)
    client.fake_os = fake_os  # type: ignore[attr-defined]
    return client


PREFIX = '/curation/projects/default/prompt_packs'


def _body() -> dict[str, object]:
    body = GENERIC_ITEM_PACK.to_dict()
    body.pop('name')
    return body


def test_p1_active_revision_pin_is_honored(app_client: TestClient) -> None:
    from src.services.labeling.vlm_prompts import active_prompt_pack, prompt_pack_stamp

    assert app_client.post(PREFIX, json={'name': 'my_pack', 'body': _body()}).status_code == 201
    r = app_client.post(f'{PREFIX}/my_pack/activate', json={'expected_active': None})
    assert r.status_code == 200, r.text
    body2 = _body()
    body2['class_system'] = 'EDITED AFTER ACTIVATION'
    r = app_client.put(f'{PREFIX}/my_pack', json={'expected_revision': 1, 'body': body2})
    assert r.status_code == 200, r.text
    active = app_client.get(f'{PREFIX}/active').json()
    served = active_prompt_pack()
    stamp = prompt_pack_stamp(served)
    print(
        'ACTIVE REF', active['active'], 'SERVED class_system', served.class_system, 'STAMP', stamp
    )
    # activation says revision 1 -> the served body must be revision 1's
    assert active['active']['revision'] == 1
    assert served.class_system == GENERIC_ITEM_PACK.class_system, (
        f'served rev-2 body while activation+stamp say rev 1 ({stamp})'
    )


def test_p2_put_cannot_create_a_reserved_name(app_client: TestClient) -> None:
    r = app_client.put(f'{PREFIX}/off', json={'expected_revision': 0, 'body': _body()})
    print('PUT off ->', r.status_code)
    assert r.status_code in (404, 422), r.text


def test_p2b_put_cannot_create_an_invalid_name(app_client: TestClient) -> None:
    r = app_client.put(f'{PREFIX}/Not_A_Slug', json={'expected_revision': 0, 'body': _body()})
    print('PUT Not_A_Slug ->', r.status_code)
    assert r.status_code in (404, 422), r.text


def test_p2c_clone_cannot_create_a_reserved_name(app_client: TestClient) -> None:
    r = app_client.post(f'{PREFIX}/{GENERIC_ITEM_PACK.name}/clone', json={'new_name': 'default'})
    print('clone -> default', r.status_code)
    assert r.status_code == 422, r.text


def test_p3_template_pack_cannot_be_activated(app_client: TestClient) -> None:
    from src.services.labeling.vlm_prompts import active_prompt_pack

    r = app_client.post(f'{PREFIX}/vehicle_wheel/activate', json={'expected_active': None})
    print('activate template ->', r.status_code, active_prompt_pack().name)
    assert r.status_code in (403, 422), r.text


def test_p5_profile_put_cannot_create_reserved_name(app_client: TestClient) -> None:
    r = app_client.put(
        '/curation/projects/default/region_profiles/none',
        json={'expected_revision': 0, 'body': {'segmenter_text_prompt': 'x region'}},
    )
    print('profile PUT none ->', r.status_code, r.text[:300])
    assert r.status_code in (404, 422), r.text


def _profile_body() -> dict[str, object]:
    from dataclasses import asdict

    from src.config import DetectionProfile

    raw = asdict(DetectionProfile(name='p'))
    raw.pop('name')
    for k, v in raw.items():
        if isinstance(v, frozenset):
            raw[k] = sorted(v)
        elif isinstance(v, tuple):
            raw[k] = list(v)
    raw.update(
        detector_model='',
        text_reader='none',
        segmenter_text_prompt='test region',
        display_name='Regions',
        display_name_singular='Region',
    )
    return raw


def test_p5b_profile_put_and_clone_bypass_name_rules(app_client, monkeypatch) -> None:
    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://nowhere:1')
    base = '/curation/projects/default/region_profiles'
    r = app_client.put(f'{base}/none', json={'expected_revision': 0, 'body': _profile_body()})
    print('profile PUT none ->', r.status_code)
    # Clone from a real stored profile (not the never-created 'none') so
    # the probe actually reaches the new_name check instead of 404ing on
    # an unknown clone source.
    created = app_client.post(base, json={'name': 'src_profile', 'body': _profile_body()})
    assert created.status_code == 201, created.text
    r2 = app_client.post(f'{base}/src_profile/clone', json={'new_name': 'Bad Name'})
    print('profile clone -> "Bad Name"', r2.status_code)
    assert r.status_code in (404, 422)
    assert r2.status_code == 422


def test_p6_from_project_clone_onto_name_unknown_to_cold_target_store(
    app_client, monkeypatch
) -> None:
    import asyncio

    from src.config.curation import base_curation_config
    from src.config.project_context import bind_project
    from src.config.projects import ProjectRecord, resources_for_new
    from src.services.config_store.index import save_config
    from src.services.config_store.store import reset_config_stores
    from src.services.projects import lifecycle as lifecycle_mod
    from src.services.projects.registry import ProjectRegistry

    fake_os = app_client.fake_os
    other = ProjectRecord(
        slug='other',
        display_name='Other',
        description='',
        status='active',
        revision=1,
        created_at='',
        updated_at='',
        origin=None,
        resources=resources_for_new('other', base_curation_config()),
    )

    async def _seed() -> None:
        from src.config import get_curation_config

        with bind_project(other):
            idx = get_curation_config().configs_index
            await save_config(
                fake_os,
                idx,
                kind='prompt_pack',
                name='other_pack',
                body={**GENERIC_ITEM_PACK.to_dict(), 'name': 'other_pack'},
                expected_revision=None,
            )
        # target ('default', bound by the test conftest) already has 'taken' in OpenSearch,
        # written by another worker process -> this process's store never saw it
        idx = get_curation_config().configs_index
        await save_config(
            fake_os, idx, kind='prompt_pack', name='taken', body=_body(), expected_revision=None
        )

    asyncio.run(_seed())
    reset_config_stores()
    monkeypatch.setattr(ProjectRegistry, 'ensure_fresh', AsyncMock(return_value=None))
    monkeypatch.setattr(lifecycle_mod, '_resolve_existing', AsyncMock(return_value=other))
    client = TestClient(app_client.app, raise_server_exceptions=False)
    r = client.post(
        f'{PREFIX}/other_pack/clone', json={'new_name': 'taken', 'from_project': 'other'}
    )
    print('cold-target from_project clone onto existing name ->', r.status_code, r.text[:120])
    assert r.status_code == 409, r.text


def test_p7_put_on_active_pack_bypasses_non_bypassable_activation_gate(
    app_client, monkeypatch
) -> None:
    import src.services.detection.profile_registry as pr
    from src.config import DetectionProfile

    multi = DetectionProfile(name='multi', max_regions_per_item=3)
    monkeypatch.setattr(pr, 'get_active_region_profile', lambda: multi)
    stripped = _body()
    stripped['combined_system'] = (
        'Return JSON keys class_id class_confidence region_visible region_bbox_correct '
        'region_confidence'
    )
    stripped['combined_batch_system'] = (
        'Return JSON results with img class_id class_confidence region_visible '
        'region_bbox_correct region_confidence'
    )
    assert app_client.post(PREFIX, json={'name': 'bad', 'body': stripped}).status_code == 201
    r_bad = app_client.post(f'{PREFIX}/bad/activate', json={'expected_active': None, 'force': True})
    print('activate stripped pack directly ->', r_bad.status_code)
    assert app_client.post(PREFIX, json={'name': 'good', 'body': _body()}).status_code == 201
    r = app_client.post(f'{PREFIX}/good/activate', json={'expected_active': None})
    print('activate good ->', r.status_code)
    r_put = app_client.put(f'{PREFIX}/good', json={'expected_revision': 1, 'body': stripped})
    from src.services.labeling.vlm_prompts import active_prompt_pack

    served = active_prompt_pack()
    print(
        'PUT stripped onto ACTIVE pack ->',
        r_put.status_code,
        '| served combined_system:',
        served.combined_system[:40],
    )
    assert r_bad.status_code == 422
    # any_domain_plan.md §4.4: "PUT ... on the active profile does NOT
    # change the active revision" -- the write itself is allowed (it
    # writes a new revision the operator can review/activate later); the
    # thing that must never happen is the stripped body going live
    # without a separate activate call re-running the for-activation gate.
    assert r_put.status_code == 200, r_put.text
    assert served.combined_system == GENERIC_ITEM_PACK.combined_system, (
        'PUT on the active pack changed what is actually running'
    )
