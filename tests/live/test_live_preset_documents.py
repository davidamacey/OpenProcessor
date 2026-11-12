"""Live: the revisioned preset documents (serving, cluster, assist-loop and near-duplicate profiles).

Each kind is saved, read, re-saved against its revision, refused when stale or misspelt, listed,
activated (the project kinds; the serving profile is deployment-wide, so a live run never activates
it) and deleted (an activated one is refused as in use). Routes and body fields come from
``contracts/openapi/curation.json`` and ``src/schemas/curation/research_config.py``. Pending the wave run.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from ._ingest_support import ingest_project
from .conftest import API_PREFIX


pytestmark = pytest.mark.live

SERVING = {
    'servers': [
        {
            'name': 'a',
            'grpc_url': 'triton-a:8001',
            'metrics_url': 'http://triton-a:8002/metrics',
            'gpu_uuids': ['GPU-00000000-0000-0000-0000-000000000000'],
        }
    ],
    'load_list': {'a': ['yolov11_small_trt_end2end']},
    'instances': {'yolov11_small_trt_end2end': 1},
    'image_prep': {'instances': 1, 'workers': 1, 'max_in_flight': 2, 'device_budget_mib': 0},
    'priorities': {'interactive': 10, 'bulk': 1},
}
CLUSTER = {
    'target': 'item',
    'level': 'item',
    'embedding_field': 'embedding',
    'method': 'minibatch_kmeans',
    'refresh': {'growth_threshold': 0.2, 'max_deferral_s': 60, 'retry_backoff_s': 30},
}
ASSIST = {'budget': {'max_vlm_crops': 10}}
NEAR_DUP: dict[str, Any] = {'version': 1}

# kind -> (path under its router, body, activatable, deployment-wide)
KINDS: dict[str, tuple[dict[str, Any], bool, bool]] = {
    'serving_profiles': (SERVING, False, True),
    'cluster_profiles': (CLUSTER, True, False),
    'assist_loop_presets': (ASSIST, True, False),
    'near_dup_profiles': (NEAR_DUP, True, False),
}


def _error(resp: Any) -> str:
    return resp.json()['detail']['error']


@pytest.mark.parametrize('path', list(KINDS))
def test_a_preset_document_goes_through_its_revisions(api: Any, opensearch: Any, path: str) -> None:
    body, activatable, global_ = KINDS[path]
    with ingest_project(api, opensearch, 'preset') as project:
        base = f'{API_PREFIX}/{path}' if global_ else f'{project.prefix}/{path}'
        name = f'live-{uuid.uuid4().hex[:6]}'
        doc = f'{base}/{name}'

        created = api.put(doc, json={'body': body, 'description': 'first'})
        assert created.status_code == 200, created.text
        assert created.json()['revision'] == 1, created.text

        again = api.put(doc, json={'body': body, 'expected_revision': None})
        assert (again.status_code, _error(again)) == (409, 'revision_conflict'), again.text

        second = api.put(doc, json={'body': body, 'description': 'second', 'expected_revision': 1})
        assert second.status_code == 200, second.text
        assert second.json()['revision'] == 2, second.text
        assert api.get(doc).json()['description'] == 'second'
        assert api.get(doc, params={'revision': 1}).json()['description'] == 'first'

        misspelt = api.put(doc, json={'body': {**body, 'not_an_option': 1}, 'expected_revision': 2})
        assert (misspelt.status_code, _error(misspelt)) == (422, 'validation_failed'), misspelt.text

        assert name in [d['name'] for d in api.get(base).json()['items']]
        assert api.get(f'{base}/no-such-{name}').status_code == 404

        if activatable:
            # Activation is compare-and-set: a project that already serves one (the near-duplicate
            # default) must be told what is active now; null only when nothing is.
            now = api.get(f'{base}/active').json()
            expected = None if now['state'] == 'none' else {k: now[k] for k in ('name', 'revision')}
            active = api.post(f'{doc}/activate', json={'expected_active': expected})
            assert active.status_code == 200, active.text
            assert (active.json()['state'], active.json()['name']) == ('active', name)
            assert api.get(f'{base}/active').json()['name'] == name
            missing = api.post(f'{base}/no-such-{name}/activate', json={})
            assert missing.status_code == 404, missing.text

        stale = api.delete(doc, params={'expected_revision': 1})
        assert (stale.status_code, _error(stale)) == (409, 'revision_conflict'), stale.text
        if activatable:
            # The activation points at it; the project's own delete removes it afterwards.
            refused = api.delete(doc, params={'expected_revision': 2})
            assert (refused.status_code, _error(refused)) == (409, 'in_use'), refused.text
            assert api.get(doc).status_code == 200
        else:
            deleted = api.delete(doc, params={'expected_revision': 2})
            assert deleted.status_code == 204, deleted.text
            assert api.get(doc).status_code == 404
