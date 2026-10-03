"""The open-vocabulary wire is fully typed: its closed value sets are published
enums with served labels, the validate issue paths have one documented format,
and the segmenter's availability is one served fact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args
from unittest.mock import AsyncMock

import httpx
import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from curation._fake_config_opensearch import FakeConfigOpenSearch
from src.services.curation import open_vocab_vocabulary as vocab
from src.services.curation.open_vocab_fields import OpenVocabStatus
from src.services.curation.reprocess_models import ReprocessFilter
from src.services.detection import profile_registry
from src.services.detection.open_vocab_select import DropReason
from src.services.detection.segmenter_gate import GateReason


PREFIX = '/curation/projects/default/open_vocab'
CONTRACT = Path(__file__).resolve().parents[2] / 'contracts' / 'openapi' / 'curation.json'


@pytest.fixture(autouse=True)
def _reset():
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()
    profile_registry._reset_registry_for_tests()
    yield
    reset_config_stores()
    profile_registry._reset_registry_for_tests()


@pytest.fixture
def segmenter(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {'status': 'ready'}

    async def _health() -> tuple[str, str | None]:
        return state['status'], None if state['status'] == 'ready' else 'down'

    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://segmenter.invalid:8000')
    monkeypatch.setattr(
        'src.routers.curation._models_segmenter.configured_segmenter_health', _health
    )
    return state


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, segmenter: dict[str, Any]) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    monkeypatch.setattr(
        'src.config.ingest_profiles.ingest_primary_profile',
        lambda: type('P', (), {'detector_model': ''})(),
    )
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    fake = FakeConfigOpenSearch()
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        'display_name': 'Street',
        'targets': [{'prompt': 'traffic cone', 'class_name': 'traffic cone'}],
    }
    body.update(overrides)
    return body


def test_every_value_of_every_closed_set_has_a_label_and_no_label_is_stale() -> None:
    served = vocab.open_vocab_vocabulary()
    assert [r['value'] for r in served['statuses']] == list(get_args(OpenVocabStatus))
    assert [r['value'] for r in served['drop_reasons']] == list(get_args(DropReason))
    assert [r['value'] for r in served['gate_reasons']] == list(get_args(GateReason))
    assert set(vocab._STATUS_LABELS) == set(get_args(OpenVocabStatus))
    assert set(vocab._DROP_REASON_LABELS) == set(get_args(DropReason))
    assert set(vocab._GATE_REASON_LABELS) == set(get_args(GateReason))
    assert all(r['label'] for rows in served.values() for r in rows)


def test_the_status_values_are_the_ones_the_pass_stamps() -> None:
    assert set(get_args(OpenVocabStatus)) == {'pending', 'done', 'skipped_gate', 'failed'}


def test_the_schema_route_serves_the_vocabulary(client: TestClient) -> None:
    body = client.get(f'{PREFIX}/schema').json()
    vocabulary = body['vocabulary']
    assert [r['value'] for r in vocabulary['statuses']] == [
        'pending',
        'done',
        'skipped_gate',
        'failed',
    ]
    assert {r['value'] for r in vocabulary['drop_reasons']} == {
        'below_min_score',
        'too_small',
        'too_large',
        'nms',
        'over_max',
        'cross_target_nms',
        'agree_existing',
        'skipped_locked',
    }
    assert {r['value'] for r in vocabulary['gate_reasons']} == {
        'disabled',
        'no_parent_class',
        'vlm_no',
        'hit_rate',
    }


def test_the_reprocess_filter_only_takes_a_real_status() -> None:
    assert ReprocessFilter(open_vocab_status=['pending', 'skipped_gate']).open_vocab_status == [
        'pending',
        'skipped_gate',
    ]
    with pytest.raises(ValidationError):
        ReprocessFilter(open_vocab_status=['failed_unavailable'])


def test_the_contract_publishes_the_enums() -> None:
    schemas = json.loads(CONTRACT.read_text())['components']['schemas']

    def enum_of(prop: dict[str, Any]) -> set[str]:
        options = prop.get('anyOf') or [prop]
        found: set[str] = set()
        for option in options:
            node = option.get('items', option)
            found |= set(node.get('enum', []))
        return found

    assert enum_of(schemas['ReprocessFilter']['properties']['open_vocab_status']) == set(
        get_args(OpenVocabStatus)
    )
    assert enum_of(schemas['OpenVocabTestHit']['properties']['drop_reason']) == set(
        get_args(DropReason)
    )
    assert enum_of(schemas['OpenVocabTestGate']['properties']['reason']) == set(
        get_args(GateReason)
    )


def test_validate_issue_fields_are_dotted_paths_with_bracketed_indexes(
    client: TestClient,
) -> None:
    targets = [
        {'prompt': 'cone', 'class_name': 'cone'},
        {'prompt': 'cone', 'class_name': 'cone'},
        {'prompt': 'cup', 'class_name': 'cup', 'min_score': 7},
    ]
    gating = {'tier3_hit_rate': {'enabled': True, 'window': 0}}
    r = client.post(
        f'{PREFIX}/validate',
        json={'name': 'paths', 'body': _body(targets=targets, gating=gating)},
    )
    fields = {e['field'] for e in r.json()['errors']}
    assert 'targets[2].min_score' in fields, 'a target field: targets[<index>].<field>'
    assert 'targets[1]' in fields, 'a whole-target issue names the target'
    assert 'gating.tier3_hit_rate.window' in fields, 'a gating field is dotted from the body root'


def test_the_issue_field_format_is_documented_in_the_contract() -> None:
    schemas = json.loads(CONTRACT.read_text())['components']['schemas']
    description = schemas['ValidationIssue']['properties']['field']['description']
    assert 'targets[2].prompt' in description
    assert 'gating.tier3_hit_rate.window' in description


def test_validate_has_a_typed_response() -> None:
    operation = json.loads(CONTRACT.read_text())['paths'][
        '/curation/projects/{project}/open_vocab/validate'
    ]['post']
    schema = operation['responses']['200']['content']['application/json']['schema']
    assert schema == {'$ref': '#/components/schemas/ValidationReport'}


@pytest.mark.parametrize(
    ('state', 'configured', 'reachable'),
    [('ready', True, True), ('unavailable', True, False)],
)
def test_the_list_serves_the_segmenter_fact(
    client: TestClient, segmenter: dict[str, Any], state: str, configured: bool, reachable: bool
) -> None:
    segmenter['status'] = state
    assert client.get(PREFIX).json()['segmenter'] == {
        'configured': configured,
        'reachable': reachable,
    }


def test_an_unconfigured_segmenter_is_never_reachable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('OP_SEGMENTER_URL')
    assert client.get(PREFIX).json()['segmenter'] == {'configured': False, 'reachable': False}


def test_the_image_id_of_the_trial_is_documented_as_the_item_wire_image_id() -> None:
    schemas = json.loads(CONTRACT.read_text())['components']['schemas']
    description = schemas['OpenVocabTestRequest']['properties']['image_id']['description']
    assert 'image_id' in description
    assert '/images/{image_id}/reprocess' in description


def _patch_http(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    import src.routers.curation.models as models_mod

    class _Client(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs['transport'] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(models_mod.httpx, 'AsyncClient', _Client)


@pytest.mark.usefixtures('vlm_env')
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('url', 'segmenter_loaded', 'status'),
    [
        ('http://segmenter:8000', True, 'ready'),
        ('http://segmenter:8000', None, 'unavailable'),
        ('', None, 'not_configured'),
    ],
)
async def test_models_status_lists_a_segmenter_row_with_no_region_profile(
    monkeypatch: pytest.MonkeyPatch, url: str, segmenter_loaded: bool | None, status: str
) -> None:
    """The row used to exist only when a region profile named a segmenter, so an
    open-vocabulary-only project looked like it had none."""
    import src.routers.curation.models as models_mod

    assert profile_registry.get_active_region_profile() is None
    monkeypatch.setenv('OP_SEGMENTER_URL', url)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == 'segmenter' and segmenter_loaded:
            return httpx.Response(200, json={'status': 'healthy', 'loaded': True})
        raise httpx.ConnectError('refused', request=request)

    _patch_http(monkeypatch, handler)
    rows = [m for m in (await models_mod.models_status())['models'] if m['kind'] == 'external']
    assert len(rows) == 1
    assert rows[0]['name'] == 'sam3'
    assert rows[0]['status'] == status
    assert rows[0]['unloadable'] is False
