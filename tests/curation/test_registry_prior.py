"""#61 item 2: optional per-pack registry prior for the VLM labeling prompt."""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation._fake_config_opensearch import FakeConfigOpenSearch
from curation.test_proposal_denylist import _labeler
from src.services.curation.registry_prior_source import (
    RegistryPriorUnavailableError,
    load_registry_prior,
)
from src.services.labeling.registry_prior import (
    MAX_REGISTRY_PRIOR_TOP_K,
    RegistryPrior,
    rank_registry_prior,
)
from src.services.labeling.vlm_models import ItemCrop
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, PromptPack


PREFIX = '/curation/projects/default/prompt_packs'


def test_rank_orders_by_validated_count_then_name_and_bounds() -> None:
    prior = rank_registry_prior(
        ['car', 'bus', 'truck', 'bike'],
        {'truck': 9, 'car': 9, 'bus': 1},
        {'forklift': 3, 'bus': 8, 'crane': 3},
        top_k=3,
    )
    assert prior == RegistryPrior(
        classes=('car', 'truck', 'bus'),
        # 'bus' is a registry name, so it is not a pending candidate
        pending=('crane', 'forklift'),
    )


def test_rank_is_none_when_off_or_empty() -> None:
    assert rank_registry_prior(['car'], {}, {}, top_k=0) is None
    assert rank_registry_prior([], {}, {}, top_k=5) is None


def test_default_pack_has_prior_off_and_round_trips() -> None:
    assert GENERIC_ITEM_PACK.registry_prior_top_k == 0
    pack = dataclasses.replace(GENERIC_ITEM_PACK, registry_prior_top_k=7)
    assert PromptPack.from_dict(pack.to_dict()).registry_prior_top_k == 7


class _Os:
    def __init__(self, resp: dict[str, Any] | Exception) -> None:
        self.resp = resp
        self.bodies: list[dict[str, Any]] = []

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        assert index
        self.bodies.append(body)
        if isinstance(self.resp, Exception):
            raise self.resp
        return self.resp


def _aggs(validated: dict[str, int], pending: dict[str, int]) -> dict[str, Any]:
    def node(counts: dict[str, int]) -> dict[str, Any]:
        return {'by_name': {'buckets': [{'key': k, 'doc_count': v} for k, v in counts.items()]}}

    return {'aggregations': {'validated': node(validated), 'pending': node(pending)}}


@pytest.mark.asyncio
async def test_source_ranks_served_counts_and_excludes_holdout() -> None:
    os_ = _Os(_aggs({'bus': 2, 'car': 7}, {'forklift': 4}))
    prior = await load_registry_prior(os_, top_k=2, registry_names=['bus', 'car', 'van'])
    assert prior == RegistryPrior(classes=('car', 'bus'), pending=('forklift',))
    assert {'term': {'test_holdout': True}} in os_.bodies[0]['query']['bool']['must_not']


@pytest.mark.asyncio
async def test_prior_or_error_reports_instead_of_raising() -> None:
    from src.services.curation.registry_prior_source import prior_or_error

    on = dataclasses.replace(GENERIC_ITEM_PACK, registry_prior_top_k=3)
    prior, err = await prior_or_error(_Os(RuntimeError('down')), on, ['car'])
    assert prior is None
    assert err is not None
    assert 'down' in err
    prior, err = await prior_or_error(_Os(_aggs({'car': 1}, {})), on, ['car'])
    assert err is None
    assert prior is not None
    assert prior.classes == ('car',)


@pytest.mark.asyncio
async def test_source_off_does_not_query() -> None:
    os_ = _Os(_aggs({}, {}))
    assert await load_registry_prior(os_, top_k=0, registry_names=['car']) is None
    assert os_.bodies == []


@pytest.mark.asyncio
async def test_source_fails_closed_when_counts_unreadable() -> None:
    with pytest.raises(RegistryPriorUnavailableError):
        await load_registry_prior(_Os(RuntimeError('down')), top_k=3, registry_names=['car'])


def _sent_text(labeler: Any) -> str:
    payload = labeler._client.post.await_args.kwargs['json']
    return payload['messages'][1]['content'][0]['text']


@pytest.mark.asyncio
async def test_prompt_carries_the_prior_and_off_does_not() -> None:
    entry = {'img': 1, 'class': 'car', 'confidence': 'high', 'proposed_class': ''}
    crops = [ItemCrop(img_id='c1', jpeg_bytes=b'j')]
    prior = RegistryPrior(classes=('car', 'bus'), pending=('forklift',))

    on = _labeler([entry])
    await on.label_or_propose_batch(crops, ['car', 'bus', 'van'], registry_prior=prior)
    text = _sent_text(on)
    assert 'Most established registry classes: car, bus.' in text
    assert 'Pending proposed classes: forklift.' in text

    off = _labeler([entry])
    await off.label_or_propose_batch(crops, ['car', 'bus', 'van'])
    assert 'registry classes' not in _sent_text(off)


@pytest.mark.asyncio
async def test_answer_outside_the_candidates_is_still_accepted() -> None:
    entry = {'img': 1, 'class': 'van', 'confidence': 'high', 'proposed_class': ''}
    labeler = _labeler([entry])
    prior = RegistryPrior(classes=('car',), pending=())
    preds = await labeler.label_or_propose_batch(
        [ItemCrop(img_id='c1', jpeg_bytes=b'j')], ['car', 'van'], registry_prior=prior
    )
    assert preds[0].class_name == 'van'
    assert preds[0].failure is None


# --- pack REST: schema, validation, round trip -------------------------------


@pytest.fixture(autouse=True)
def _reset_caches():
    from src.services.config_store.store import reset_config_stores

    reset_config_stores()
    yield
    reset_config_stores()


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    fake_os = FakeConfigOpenSearch()
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    return TestClient(app)


def _body(**over: object) -> dict[str, object]:
    body = GENERIC_ITEM_PACK.to_dict()
    body.pop('name')
    body.update(over)
    return body


def test_schema_serves_the_setting_as_an_int_row(app_client: TestClient) -> None:
    fields = {f['field']: f for f in app_client.get(f'{PREFIX}/schema').json()['fields']}
    assert fields['registry_prior_top_k']['kind'] == 'int'
    assert fields['registry_prior_top_k']['group'] == 'open_classify'
    assert str(MAX_REGISTRY_PRIOR_TOP_K) in fields['registry_prior_top_k']['help']


@pytest.mark.parametrize('bad', [-1, MAX_REGISTRY_PRIOR_TOP_K + 1, True, 'ten', 2.5, None])
def test_bad_top_k_is_a_validation_error(app_client: TestClient, bad: object) -> None:
    r = app_client.post(f'{PREFIX}/validate', json={'body': _body(registry_prior_top_k=bad)})
    assert r.status_code == 200, r.text
    report = r.json()
    assert report['ok'] is False, bad
    assert any(e['field'] == 'registry_prior_top_k' for e in report['errors'])


@pytest.mark.parametrize('good', [0, 1, MAX_REGISTRY_PRIOR_TOP_K])
def test_top_k_in_range_validates_and_round_trips(app_client: TestClient, good: int) -> None:
    r = app_client.post(
        PREFIX,
        json={'name': 'prior_pack', 'description': '', 'body': _body(registry_prior_top_k=good)},
    )
    assert r.status_code == 201, r.text
    got = app_client.get(f'{PREFIX}/prior_pack').json()['body']
    assert got['registry_prior_top_k'] == good


def test_missing_top_k_defaults_off(app_client: TestClient) -> None:
    body = _body()
    body.pop('registry_prior_top_k')
    r = app_client.post(f'{PREFIX}/validate', json={'body': body})
    assert r.json()['ok'] is True


# --- wiring: POST /vlm/label_batch ------------------------------------------


def _wire_label_batch(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, pack: PromptPack, seen: list[Any]
) -> tuple[Any, Any]:
    import io
    from types import SimpleNamespace

    from PIL import Image

    import src.routers.curation.vlm as vlm_mod
    from curation.query_fakes import QueryFakeOpenSearch
    from src.clients.curation_opensearch.registry import ClassRegistry
    from src.config.curation import base_curation_config
    from src.services.labeling.vlm_client import VlmIdentity

    items = base_curation_config().items_index
    buf = io.BytesIO()
    Image.new('RGB', (32, 32), (1, 2, 3)).save(buf, format='JPEG')
    (tmp_path / 'c1.jpg').write_bytes(buf.getvalue())
    monkeypatch.setattr(
        vlm_mod, 'get_curation_config', lambda: SimpleNamespace(crop_cache_dir=tmp_path)
    )
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    reg.add_class('widget')
    monkeypatch.setattr(vlm_mod, 'get_class_registry', lambda: reg)

    async def _no_pack(_os: Any) -> None:
        return None

    monkeypatch.setattr(vlm_mod, '_default_pack_name', _no_pack)
    fake = QueryFakeOpenSearch(
        {
            items: {
                'c1': {
                    'crop_id': 'c1',
                    'image_path': '/data/c1.jpg',
                    'bbox_norm': [0.1, 0.1, 0.5, 0.5],
                    'class_source': 'item_proposal',
                    'class_validated': False,
                }
            }
        }
    )

    class _Labeler:
        identity = VlmIdentity('env@None', 'test-vlm')
        _pack = pack

        async def label_or_propose_batch(
            self, _crops: list[Any], _names: list[str], *, registry_prior: Any = None
        ) -> list[Any]:
            seen.append(registry_prior)
            return []

    monkeypatch.setattr(vlm_mod, '_get_vlm_labeler', lambda *_a, **_k: _Labeler())
    return vlm_mod, fake


@pytest.mark.usefixtures('vlm_env')
@pytest.mark.asyncio
async def test_label_batch_passes_the_served_prior_only_when_the_pack_asks(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.routers.curation.vlm import VlmLabelBatchRequest

    seen: list[Any] = []
    off = dataclasses.replace(GENERIC_ITEM_PACK, registry_prior_top_k=0)
    vlm_mod, fake = _wire_label_batch(tmp_path, monkeypatch, off, seen)
    await vlm_mod.vlm_label_batch(VlmLabelBatchRequest(crop_ids=['c1']), fake)
    assert seen == [None]

    seen.clear()
    on = dataclasses.replace(GENERIC_ITEM_PACK, registry_prior_top_k=5)
    (tmp_path / 'on').mkdir()
    vlm_mod, fake = _wire_label_batch(tmp_path / 'on', monkeypatch, on, seen)
    await vlm_mod.vlm_label_batch(VlmLabelBatchRequest(crop_ids=['c1']), fake)
    assert seen[0] is not None
    assert seen[0].classes == ('widget',)


@pytest.mark.usefixtures('vlm_env')
@pytest.mark.asyncio
async def test_label_batch_refuses_when_the_prior_cannot_be_built(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    from src.routers.curation.vlm import VlmLabelBatchRequest

    seen: list[Any] = []
    on = dataclasses.replace(GENERIC_ITEM_PACK, registry_prior_top_k=5)
    vlm_mod, fake = _wire_label_batch(tmp_path, monkeypatch, on, seen)
    fake.search = AsyncMock(side_effect=RuntimeError('down'))
    with pytest.raises(HTTPException) as exc:
        await vlm_mod.vlm_label_batch(VlmLabelBatchRequest(crop_ids=['c1']), fake)
    assert exc.value.status_code == 503
    assert seen == []
