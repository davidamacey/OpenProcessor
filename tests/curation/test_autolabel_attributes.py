"""Pack attributes in a whole-project VLM run (`vlm_attributes`): the auto-label VLM stage
asks the pack's combined `assist` call and stores the attributes in each item's `assist`
slot, apart from the class."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from curation.conftest import SCOPED
from curation.query_fakes import QueryFakeOpenSearch
from curation.test_vlm_empty_class_answer import ITEMS, _item
from src.services.labeling.vlm_attributes import AttributeSpec
from src.services.labeling.vlm_client import VlmIdentity
from src.services.labeling.vlm_models import AttributeValue, VlmClassPrediction
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


pytestmark = pytest.mark.usefixtures('vlm_env_active')

PACK = dataclasses.replace(
    GENERIC_ITEM_PACK,
    name='cars',
    attributes=[AttributeSpec(key='color', values=['red', 'blue'], applies_to='item')],
)


class _Labeler:
    identity = VlmIdentity('env@None', 'test-vlm')
    model = 'fake-vlm'
    _pack = PACK

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def label_or_propose_batch(self, crops: list[Any], *_a: Any, **_k: Any) -> list[Any]:
        self.calls.append('classify')
        return [
            VlmClassPrediction(img_id=c.img_id, class_name='sportscar', confidence='high')
            for c in crops
        ]

    async def assist_batch(self, crops: list[Any], *_a: Any, **_k: Any) -> list[Any]:
        self.calls.append('assist')
        return [
            VlmClassPrediction(
                img_id=c.img_id,
                class_name='sportscar',
                confidence='high',
                attributes={'color': AttributeValue(value='red', confidence='high')},
            )
            for c in crops
        ]


async def _run(monkeypatch: pytest.MonkeyPatch, *, vlm_attributes: bool) -> Any:
    from curation.test_pipeline import _FakeClassEntry, _FakeRegistry
    from src.routers.curation.pipeline_public import pipeline_auto_label
    from src.services.curation import auto_label_run, image_serving
    from src.services.curation.autolabel import selection, vlm_stage

    fake = QueryFakeOpenSearch({ITEMS: {cid: _item(cid) for cid in ('a', 'b')}})
    labeler = _Labeler()
    monkeypatch.setattr(selection, 'classifier_class_sources', lambda: frozenset({'det_model'}))
    monkeypatch.setattr(vlm_stage, 'get_vlm_labeler', lambda *_a, **_k: labeler)
    monkeypatch.setattr(vlm_stage, 'resolve_pack', lambda *_a, **_k: PACK)
    monkeypatch.setattr(
        auto_label_run, 'resolve_run_prompt_pack', AsyncMock(return_value=(None, None))
    )
    monkeypatch.setattr(auto_label_run, 'pipeline_health_snapshot', AsyncMock(return_value={}))
    monkeypatch.setattr(
        'src.clients.curation_opensearch.registry.get_class_registry',
        lambda: _FakeRegistry([_FakeClassEntry(7, 'sportscar')]),
    )
    monkeypatch.setattr(
        'src.services.labeling.vlm_class_names.format_class_catalog', lambda *_a, **_k: ''
    )
    monkeypatch.setattr(image_serving, 'resolve_crop_root', lambda _p: '/data')
    monkeypatch.setattr(image_serving, 'resolve_safe_path', lambda p, _r: p)
    monkeypatch.setattr(
        image_serving,
        'THUMBNAIL_CACHE',
        SimpleNamespace(get_or_compute=lambda *_a, **_k: b'jpeg'),
    )
    result = await pipeline_auto_label(
        opensearch=fake, train_clusters=False, run_vlm=True, vlm_attributes=vlm_attributes
    )
    return result, fake, labeler


@pytest.mark.asyncio
async def test_a_run_with_attributes_asks_the_combined_call_and_stores_the_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, fake, labeler = await _run(monkeypatch, vlm_attributes=True)

    assert labeler.calls == ['assist']
    assert result['stages']['vlm']['updated'] == 2
    for cid in ('a', 'b'):
        doc = fake.store[ITEMS][cid]
        assert doc['class_name'] == 'sportscar'
        slot = doc['assist']
        assert slot['attributes'] == [{'key': 'color', 'value': 'red', 'confidence': 'high'}]
        assert slot['class_name'] == 'sportscar'
        assert slot['prompt_pack'] == 'cars'
        assert slot['run_id'].startswith('al_')


@pytest.mark.asyncio
async def test_a_run_without_attributes_keeps_the_classify_call_and_writes_no_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _result, fake, labeler = await _run(monkeypatch, vlm_attributes=False)

    assert labeler.calls == ['classify']
    assert 'assist' not in fake.store[ITEMS]['a']


def test_start_refuses_attributes_without_the_vlm_stage(stack: Any) -> None:
    r = stack.client.post(f'{SCOPED}/pipeline/auto_label/start?vlm_attributes=true')
    assert (r.status_code, r.json()['detail']['error']) == (422, 'invalid_request')


def test_start_refuses_a_pack_that_declares_no_item_attributes(stack: Any) -> None:
    r = stack.client.post(f'{SCOPED}/pipeline/auto_label/start?run_vlm=true&vlm_attributes=true')
    assert (r.status_code, r.json()['detail']['error']) == (422, 'pack_declares_no_attributes')
