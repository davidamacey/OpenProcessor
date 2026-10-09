"""Switching a project's VLM takes effect on the next call, with no restart,
and what an item records is the endpoint that answered it (W9.3). Two fake
upstreams stand behind real ``httpx`` transports; the label route is the
real one, over an in-memory item store."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from PIL import Image

from curation.conftest import ACTIVE, SCOPED, HybridOpenSearch, good_probe
from curation.query_fakes import QueryFakeOpenSearch
from src.clients.curation_opensearch.registry import ClassRegistry
from src.config.curation import base_curation_config


ITEMS = base_curation_config().items_index
LABEL = f'{SCOPED}/vlm/label_batch'


class Upstreams:
    """``a.vlm.test`` and ``b.vlm.test``: OpenAI-shaped, labelling every
    crop ``widget``. Records which host got which model id."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []  # (host, model sent)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host not in ('a.vlm.test', 'b.vlm.test'):
            raise httpx.ConnectError(f'no network to {host}', request=request)
        if not request.url.path.endswith('/chat/completions'):
            return httpx.Response(200, json={'data': [{'id': f'served-{host[0]}'}]})
        payload = json.loads(request.content)
        self.calls.append((host, payload['model']))
        images = sum(
            1
            for part in payload['messages'][-1]['content']
            if isinstance(part, dict) and part.get('type') == 'image_url'
        )
        results = [
            {'img': i, 'class': 'widget', 'confidence': 'high'} for i in range(1, images + 1)
        ]
        return httpx.Response(
            200, json={'choices': [{'message': {'content': json.dumps({'results': results})}}]}
        )


@pytest.fixture
def stack(vlm_api, tmp_path, monkeypatch: pytest.MonkeyPatch, reference_region_profile):
    upstream = Upstreams()

    async def serve(_transport: Any, request: httpx.Request) -> httpx.Response:
        return await upstream.handle(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', serve)

    vlm_api.dns['a.vlm.test'] = ['10.0.0.1']
    vlm_api.dns['b.vlm.test'] = ['10.0.0.2']
    buf = io.BytesIO()
    Image.new('RGB', (32, 32), (1, 2, 3)).save(buf, format='JPEG')
    docs = {}
    for cid in ('c1', 'c2', 'c3', 'c4'):
        (tmp_path / f'{cid}.jpg').write_bytes(buf.getvalue())
        docs[cid] = {
            'crop_id': cid,
            'image_path': f'/data/{cid}.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_source': 'item_proposal',
            'class_validated': False,
        }
    items = QueryFakeOpenSearch({ITEMS: docs})
    monkeypatch.setenv('OP_CROP_CACHE_DIR', str(tmp_path))
    import src.config.curation as curation_config_mod
    import src.routers.curation.vlm as vlm_mod

    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
    monkeypatch.setattr(
        vlm_mod, 'get_curation_config', lambda: SimpleNamespace(crop_cache_dir=tmp_path)
    )
    registry = ClassRegistry(path=tmp_path / 'class_registry.json')
    registry.add_class('widget')
    monkeypatch.setattr(vlm_mod, 'get_class_registry', lambda: registry)

    from src.routers.curation import _raw_opensearch_dep

    hybrid = HybridOpenSearch(vlm_api.fake_os, items)
    vlm_api.client.app.dependency_overrides[_raw_opensearch_dep] = lambda: hybrid

    for name, host, root in (
        ('ua', 'a.vlm.test', 'org/a-root'),
        ('ub', 'b.vlm.test', 'org/b-root'),
    ):
        vlm_api.probe_record[0] = good_probe(root=root)
        vlm_api.ready(name, base_url=f'http://{host}/v1', model=f'alias-{name[1]}')
    return SimpleNamespace(api=vlm_api, upstream=upstream, items=items)


def _label(stack, crop: str, **params: Any) -> Any:
    response = stack.api.client.post(LABEL, params=params, json={'crop_ids': [crop]})
    assert response.status_code == 200, response.text
    return response.json()


def _doc(stack, crop: str) -> dict[str, Any]:
    return stack.items.docs(ITEMS)[crop]


def test_the_next_call_goes_to_the_newly_activated_endpoint_and_is_stamped_by_it(stack) -> None:
    api = stack.api
    assert api.activate('ua', expected_active=None).status_code == 200
    _label(stack, 'c1')
    assert stack.upstream.calls == [('a.vlm.test', 'alias-a')]  # the alias is what is SENT
    assert (_doc(stack, 'c1')['vlm_endpoint'], _doc(stack, 'c1')['vlm_model']) == (
        'ua@1',
        'org/a-root',  # what is RECORDED is the served model's root
    )

    assert api.activate('ub', expected_active={'name': 'ua', 'revision': 1}).status_code == 200
    _label(stack, 'c2')  # no restart, no reload: the very next call
    assert stack.upstream.calls[-1] == ('b.vlm.test', 'alias-b')
    assert (_doc(stack, 'c2')['vlm_endpoint'], _doc(stack, 'c2')['vlm_model']) == (
        'ub@1',
        'org/b-root',
    )
    # the earlier item keeps the endpoint that answered it
    assert _doc(stack, 'c1')['vlm_endpoint'] == 'ua@1'


def test_a_per_run_selection_uses_that_endpoint_for_that_call_only(stack) -> None:
    api = stack.api
    assert api.activate('ua', expected_active=None).status_code == 200
    _label(stack, 'c1', vlm='ub')
    assert stack.upstream.calls == [('b.vlm.test', 'alias-b')]
    assert _doc(stack, 'c1')['vlm_endpoint'] == 'ub@1'
    assert api.active()['active'] == {'name': 'ua', 'revision': 1}  # settings untouched
    _label(stack, 'c2')
    assert stack.upstream.calls[-1] == ('a.vlm.test', 'alias-a')


def test_a_pinned_revision_is_used_even_after_the_endpoint_is_edited(stack) -> None:
    api = stack.api
    saved = api.client.put(
        '/curation/vlm/endpoints/ua',
        json={
            'expected_revision': 1,
            'body': api.body(base_url='http://b.vlm.test/v1', model='alias-edited'),
        },
    )
    assert saved.status_code == 200, saved.text
    _label(stack, 'c1', vlm='ua@1')
    assert stack.upstream.calls == [('a.vlm.test', 'alias-a')]  # revision 1's URL, not 2's
    assert _doc(stack, 'c1')['vlm_endpoint'] == 'ua@1'


def test_with_the_vlm_off_the_route_refuses_and_writes_nothing(stack) -> None:
    api = stack.api
    assert api.activate('ua', expected_active=None).status_code == 200
    off = api.client.post(
        f'{ACTIVE}/deactivate', json={'expected_active': {'name': 'ua', 'revision': 1}}
    )
    assert off.status_code == 200
    response = api.client.post(LABEL, json={'crop_ids': ['c1']})
    assert response.status_code == 409
    assert response.json()['detail']['error'] == 'vlm_not_configured'
    assert stack.upstream.calls == []
    assert 'vlm_endpoint' not in _doc(stack, 'c1')


@pytest.mark.asyncio
async def test_a_started_job_keeps_the_endpoint_it_was_accepted_against(stack, monkeypatch) -> None:
    """``/pipeline/auto_label/start`` pins ``(name, revision)`` into the job;
    later activations and edits change nothing for it, and the job resolves
    that immutable revision by id (in whatever process runs it)."""
    from src.routers.curation.pipeline_vlm import job_endpoint
    from src.services.curation.autolabel import job as auto_label_job

    api = stack.api
    captured: dict[str, Any] = {}

    def fake_start(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        captured.update(kwargs)
        return {'status': 'queued', 'job_id': 'j1'}

    monkeypatch.setattr(auto_label_job, 'start_job', fake_start)
    started = api.client.post(
        f'{SCOPED}/pipeline/auto_label/start', params={'run_vlm': 'true', 'vlm': 'ua'}
    )
    assert started.status_code == 200, started.text
    assert captured['vlm_endpoint'] == 'ua'
    assert captured['vlm_endpoint_revision'] == 1
    assert captured['vlm_resolved'] is True

    # meanwhile: the project moves to b, and a gets a new revision
    assert api.activate('ub', expected_active=None).status_code == 200
    api.client.put(
        '/curation/vlm/endpoints/ua',
        json={'expected_revision': 1, 'body': api.body(base_url='http://b.vlm.test/v1', model='x')},
    )
    resolved = await job_endpoint(
        api.fake_os,
        vlm=None,
        acknowledge_external=False,
        pack=None,
        pinned_name=captured['vlm_endpoint'],
        pinned_revision=captured['vlm_endpoint_revision'],
        resolved=captured['vlm_resolved'],
    )
    assert resolved is not None
    assert resolved.ref == 'ua@1'
    assert resolved.body.base_url == 'http://a.vlm.test/v1'


def test_the_public_route_cannot_claim_a_job_is_already_resolved(stack) -> None:
    """``vlm_resolved`` / ``vlm_endpoint*`` are internal job arguments: not
    query parameters of any route."""
    from src.main import app

    for route in app.routes:
        params = {p.name for p in getattr(getattr(route, 'dependant', None), 'query_params', [])}
        assert not params & {'vlm_resolved', 'vlm_endpoint', 'vlm_endpoint_revision'}, route.path
