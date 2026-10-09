"""Every way a VLM endpoint can be selected goes through ONE gate.

The W3/W4 lesson (a gate bypass found four times because each caller
re-implemented the checks): a single test walks every selection path with an
input that has to be refused. Here that input is an endpoint that passed
validation when it was saved and whose host has since started resolving to
the cloud metadata address (a DNS-rebinding-shaped change). If any path
skipped the shared gate, it would accept it.

``test_every_route_with_a_vlm_parameter_is_walked`` keeps the list honest:
a new route that takes ``?vlm=`` must be added to :data:`RUN_ROUTES`.
"""

from __future__ import annotations

import asyncio
import base64
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _route_helpers import api_routes

import src.services.labeling.vlm_url_policy as policy
from curation.conftest import ACTIVE, GLOBAL_VLM, SCOPED
from src.config.curation import IndexRole, base_curation_config
from src.config.project_context import bind_project
from src.config.projects import ProjectRecord, resources_for_new
from src.services.labeling.vlm_endpoint_body import VlmEndpointBody


REBOUND = 'rebound'
DENIED = 'vlm_url_denied_address'
IMG = base64.b64encode(b'\xff\xd8\xff\xd9').decode()

#: The routes that accept ``?vlm=`` and the request each one needs.
RUN_ROUTES: dict[tuple[str, str], dict[str, Any]] = {
    ('POST', '/vlm/label_batch'): {'json': {'crop_ids': ['c1']}},
    ('POST', '/vlm/verify_regions'): {'json': {'crop_ids': ['c1']}},
    ('POST', '/vlm/verify_region_batch'): {
        'json': {'items': [{'crop_id': 'c1', 'region_image_b64': IMG}]}
    },
    ('POST', '/vlm/region_visible_batch'): {
        'json': {'items': [{'crop_id': 'c1', 'image_b64': IMG}]}
    },
    ('POST', '/vlm/label_cluster/{cluster_id}'): {},
    ('POST', '/pipeline/auto_label'): {},
    ('POST', '/pipeline/auto_label/start'): {},
}


#: The test-on-crop routes name an endpoint in the BODY (``vlm_name`` +
#: ``vlm_revision`` / ``vlm_draft``), not with ``?vlm=``; the request each one
#: needs to reach the VLM gate.
TEST_ROUTES: dict[tuple[str, str], dict[str, Any]] = {
    ('POST', '/prompt_packs/test'): {'call': 'classify', 'crop_ids': ['c1']},
    ('POST', '/region_profiles/test'): {
        'crop_id': 'c1',
        'verify': True,
        'draft': {
            'detector_model': '',
            'text_reader': 'none',
            'text_hint_enabled': False,
            'ocr_pipeline_model': '',
            'segmenter_text_prompt': 'wheel',
            'region_class_name': 'wheel',
            'display_name': 'Wheels',
            'display_name_singular': 'Wheel',
        },
    },
}


@pytest.fixture
def rebound(vlm_api, reference_region_profile):
    """A good stored endpoint (``rebound``), then the DNS answer turns bad."""
    vlm_api.dns['rebound.example.com'] = ['10.0.0.7']
    vlm_api.ready('good')
    vlm_api.ready(REBOUND, base_url='http://rebound.example.com/v1')
    assert vlm_api.activate('good', expected_active=None).status_code == 200
    vlm_api.dns['rebound.example.com'] = ['169.254.169.254']
    policy.reset_policy_caches()
    return vlm_api


def _detail(response: Any) -> dict[str, Any]:
    return response.json()['detail']


def _refused_for_the_address(response: Any) -> None:
    assert response.status_code == 422, response.text
    detail = _detail(response)
    codes = [e['code'] for e in detail['report']['errors']]
    assert DENIED in codes, detail
    assert all(not e['bypassable'] for e in detail['report']['errors'] if e['code'] == DENIED)


def _project_writes(api) -> int:
    return len(api.fake_os._docs.get('op_prj_default__configs', {}))


# ---- the walk ---------------------------------------------------------------


def test_activate_refuses(rebound) -> None:
    before = rebound.active()
    _refused_for_the_address(
        rebound.activate(REBOUND, expected_active={'name': 'good', 'revision': 1}, force=True)
    )
    assert rebound.active()['active'] == before['active']


def test_rollback_refuses(rebound) -> None:
    rebound.ready('third')
    assert (
        rebound.activate('third', expected_active={'name': 'good', 'revision': 1}).status_code
        == 200
    )
    doc = rebound.fake_os._docs['op_prj_default__configs']['activation:vlm']['_source']
    doc['previous'] = {'name': REBOUND, 'revision': 1}
    response = rebound.client.post(
        f'{ACTIVE}/active/rollback', json={'expected_active': {'name': 'third', 'revision': 1}}
    )
    _refused_for_the_address(response)
    assert rebound.active()['active']['name'] == 'third'


def test_put_settings_refuses(rebound) -> None:
    response = rebound.client.put(f'{SCOPED}/settings', json={'defaults': {'vlm': REBOUND}})
    assert response.status_code == 422, response.text
    assert rebound.active()['active']['name'] == 'good'
    assert 'rebound' not in str(
        rebound.fake_os._docs.get('op_prj_default__configs', {}).get('activation:vlm')
    )


def test_put_settings_gates_the_vlm_together_with_the_pack_and_profile(rebound) -> None:
    """A combined request cannot smuggle the bad endpoint past the gate by
    also changing another axis."""
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    response = rebound.client.put(
        f'{SCOPED}/settings',
        json={'defaults': {'vlm': REBOUND, 'prompt_pack': GENERIC_ITEM_PACK.name}},
    )
    assert response.status_code == 422, response.text
    assert rebound.active()['active']['name'] == 'good'


@pytest.mark.parametrize('route', sorted(RUN_ROUTES))
def test_per_run_selection_refuses(rebound, route: tuple[str, str]) -> None:
    method, template = route
    url = SCOPED + template.replace('{cluster_id}', '1')
    kwargs = RUN_ROUTES[route]
    good = rebound.lenient.request(method, url, params={'vlm': 'good'}, **kwargs)
    assert good.status_code != 422, 'the positive control must get past the VLM gate: ' + good.text
    bad = rebound.lenient.request(method, url, params={'vlm': REBOUND}, **kwargs)
    _refused_for_the_address(bad)
    # a pinned revision is refused the same way
    pinned = rebound.lenient.request(method, url, params={'vlm': f'{REBOUND}@1'}, **kwargs)
    _refused_for_the_address(pinned)


@pytest.mark.parametrize('route', sorted(RUN_ROUTES))
def test_per_run_selection_of_an_unknown_endpoint_is_a_422(rebound, route: tuple[str, str]) -> None:
    method, template = route
    url = SCOPED + template.replace('{cluster_id}', '1')
    response = rebound.lenient.request(method, url, params={'vlm': 'nope@3'}, **RUN_ROUTES[route])
    assert response.status_code == 422, response.text
    assert _detail(response)['error'] == 'unknown_vlm'
    assert 'good' in _detail(response)['valid_ids']


@pytest.mark.parametrize('route', sorted(TEST_ROUTES))
def test_test_on_crop_selection_refuses(rebound, monkeypatch, route: tuple[str, str]) -> None:
    """A test run sends a real crop to the endpoint it names, so every way of
    naming one -- by name, by pinned revision, as an unsaved draft -- is
    refused for the rebound address, before any crop is read."""
    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://seg.test:8000')
    from curation.conftest import HybridOpenSearch
    from curation.query_fakes import QueryFakeOpenSearch
    from src.routers.curation import _raw_opensearch_dep

    hybrid = HybridOpenSearch(rebound.fake_os, QueryFakeOpenSearch({}))
    rebound.client.app.dependency_overrides[_raw_opensearch_dep] = lambda: hybrid
    method, template = route
    url = SCOPED + template
    base = TEST_ROUTES[route]

    def call(**vlm: Any) -> Any:
        return rebound.lenient.request(method, url, json={**base, **vlm})

    good = call(vlm_name='good')
    assert good.status_code != 422, 'the positive control must get past the VLM gate: ' + good.text
    assert good.status_code == 404, good.text  # on to the crop lookup (there is no item store)
    _refused_for_the_address(call(vlm_name=REBOUND))
    _refused_for_the_address(call(vlm_name=REBOUND, vlm_revision=1))
    _refused_for_the_address(
        call(vlm_draft={'base_url': 'http://rebound.example.com/v1', 'model': 'm'})
    )


def test_every_route_naming_an_endpoint_in_its_body_is_walked() -> None:
    from src.main import app

    found: set[tuple[str, str]] = set()
    for ctx in api_routes(app):
        route = ctx.original_route
        if '/projects/{project}' not in ctx.path:
            continue
        field = getattr(route, 'body_field', None)
        model = field.field_info.annotation if field is not None else None
        if 'vlm_name' not in getattr(model, 'model_fields', {}):
            continue
        for method in ctx.methods - {'HEAD'}:
            found.add((method, ctx.path.split('/projects/{project}', 1)[1]))
    assert found == set(TEST_ROUTES), (
        f'routes naming a VLM endpoint in the body that this walk does not cover: '
        f'{sorted(found - set(TEST_ROUTES))}; walked but gone: {sorted(set(TEST_ROUTES) - found)}'
    )


def test_the_clone_of_a_project_refuses(rebound, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.projects import lifecycle as lifecycle_mod

    def record(slug: str) -> ProjectRecord:
        now = datetime.now(UTC).isoformat()
        return ProjectRecord(
            slug=slug,
            display_name=slug,
            description='',
            status='active',
            revision=1,
            created_at=now,
            updated_at=now,
            origin=None,
            resources=resources_for_new(slug, base_curation_config()),
        )

    source, target = record('alpha'), record('beta')
    client = rebound.fake_os
    now = '2026-01-01T00:00:00+00:00'
    client._docs.setdefault(source.resources.indexes[IndexRole.CONFIGS], {})['activation:vlm'] = {
        '_source': {
            'doc_type': 'activation',
            'axis': 'vlm',
            'name': REBOUND,
            'revision': 1,
            'activated_at': now,
            'previous': None,
            'acked_refs': {},
            'external_ack_at': None,
        },
        '_seq_no': 1,
    }
    monkeypatch.setattr(lifecycle_mod, '_resolve_existing', AsyncMock(return_value=source))

    async def clone() -> None:
        with bind_project(target):
            await lifecycle_mod.clone_settings(
                client, target_record=target, from_slug='alpha', axes=['vlm_activation']
            )

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        asyncio.run(clone())
    assert caught.value.status_code == 422
    assert DENIED in str(caught.value.detail)
    target_docs = client._docs.get(target.resources.indexes[IndexRole.CONFIGS], {})
    assert 'activation:vlm' not in target_docs


def test_a_draft_body_is_refused_and_can_never_be_activated(rebound) -> None:
    """A draft (an unsaved body under test) goes through the same gate; its
    ref is not an endpoint name, so it also cannot be activated."""
    from fastapi import HTTPException

    from src.services.config_store.vlm_gate import enforce_vlm_gate
    from src.services.labeling.vlm_endpoints import resolve_vlm_source

    body = VlmEndpointBody(base_url='http://rebound.example.com/v1', model='m')

    async def attempt() -> None:
        draft = await resolve_vlm_source(rebound.fake_os, draft=body)
        assert draft is not None
        assert draft.name == '_draft'
        await enforce_vlm_gate(draft, mode='test')

    with pytest.raises(HTTPException) as caught:
        asyncio.run(attempt())
    assert caught.value.status_code == 422
    assert DENIED in str(caught.value.detail)
    assert rebound.activate(
        '_draft', expected_active={'name': 'good', 'revision': 1}
    ).status_code in (
        404,
        422,
    )


@pytest.mark.parametrize('mode', ['settings', 'clone', 'rollback', 'run', 'test'])
def test_force_is_honoured_only_for_activation(rebound, mode: str) -> None:
    """The gate itself ignores ``force`` outside ``activate``: an unprobed
    endpoint stays refused however the caller asks."""
    from fastapi import HTTPException

    from src.services.config_store.vlm_gate import enforce_vlm_gate
    from src.services.labeling.vlm_endpoints import resolve_vlm_source

    rebound.dns['fresh.example.com'] = ['10.0.0.8']
    rebound.create('unprobed', base_url='http://fresh.example.com/v1')

    async def attempt(force_mode: str) -> Any:
        endpoint = await resolve_vlm_source(rebound.fake_os, 'unprobed')
        assert endpoint is not None
        return await enforce_vlm_gate(endpoint, mode=force_mode, force=True)  # type: ignore[arg-type]

    if mode in ('run', 'test'):
        # per-run modes do not require a probe at all
        assert asyncio.run(attempt(mode)).report.ok
        return
    with pytest.raises(HTTPException) as caught:
        asyncio.run(attempt(mode))
    codes = [e['code'] for e in _detail_of(caught.value)['report']['errors']]
    assert codes == ['vlm_not_probed']
    # only ``activate`` honours it: the same call there does not raise
    honoured = asyncio.run(attempt('activate'))
    assert [e.code for e in honoured.report.errors] == ['vlm_not_probed']


def _detail_of(exc: Any) -> dict[str, Any]:
    return exc.detail


# ---- the list is complete ---------------------------------------------------


def test_every_route_with_a_vlm_parameter_is_walked() -> None:
    from src.main import app

    found: set[tuple[str, str]] = set()
    for ctx in api_routes(app):
        route = ctx.original_route
        if '/projects/{project}' not in ctx.path:
            continue
        if 'vlm' not in {p.name for p in route.dependant.query_params}:
            continue
        for method in ctx.methods - {'HEAD'}:
            found.add((method, ctx.path.split('/projects/{project}', 1)[1]))
    assert found == set(RUN_ROUTES), (
        f'routes taking ?vlm= that this walk does not cover: {sorted(found - set(RUN_ROUTES))}; '
        f'walked but gone: {sorted(set(RUN_ROUTES) - found)}'
    )


def test_only_the_gate_runs_the_activation_validation() -> None:
    """``validate_vlm_endpoint`` is called with a non-constant
    ``for_activation`` (the strict pass) from the gate alone; every other
    caller is a read-only validate/probe route passing ``False``."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / 'src'
    strict: set[str] = set()
    callers: set[str] = set()
    for path in src.rglob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if getattr(func, 'id', getattr(func, 'attr', None)) != 'validate_vlm_endpoint':
                continue
            rel = path.relative_to(src).as_posix()
            callers.add(rel)
            flag = next((k.value for k in node.keywords if k.arg == 'for_activation'), None)
            assert flag is not None, f'{rel}: for_activation must be explicit'
            if not (isinstance(flag, ast.Constant) and flag.value is False):
                strict.add(rel)
    assert strict == {'services/config_store/vlm_gate.py'}, strict
    assert 'services/config_store/vlm_gate.py' in callers


def test_the_global_registry_routes_never_select_anything(vlm_api) -> None:
    """Creating, saving, cloning or probing an endpoint changes no project's
    activation (the registry is not a selection path)."""
    vlm_api.ready('alpha')
    vlm_api.client.put(
        f'{GLOBAL_VLM}/alpha', json={'expected_revision': 1, 'body': vlm_api.body(model='x')}
    )
    vlm_api.client.post(f'{GLOBAL_VLM}/alpha/clone', json={'new_name': 'copy'})
    assert 'activation:vlm' not in vlm_api.fake_os._docs.get('op_prj_default__configs', {})
    assert vlm_api.active()['active'] == {'name': None, 'revision': None}
