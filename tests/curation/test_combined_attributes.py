"""Item attributes in the region worker's combined call: the pack's item attributes ride on
the classify call, the reply parser keeps declared keys with allowed values, and the worker
writes them to the item's `assist` slot."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from scripts.curation.worker.stage_b import combined_slot_meta
from scripts.curation.worker.verify import _combined_class_update
from src.services.curation.assist.slot_write import RunMeta
from src.services.labeling.vlm_attributes import AttributeSpec
from src.services.labeling.vlm_client import VlmIdentity
from src.services.labeling.vlm_labeler import VlmLabeler
from src.services.labeling.vlm_models import AttributeValue, CombinedCrop, VlmCombinedReply
from src.services.labeling.vlm_prompts import GENERIC_REGION_PACK
from tests.curation.test_vlm_combined import _make_jpeg


PACK = dataclasses.replace(
    GENERIC_REGION_PACK,
    attributes=[AttributeSpec(key='color', values=['red', 'blue'], applies_to='item')],
)
REPLY = {
    'class_id': 0,
    'class_confidence': 'high',
    'region_visible': True,
    'region_boxes': [{'box': 1, 'region_bbox_correct': True}],
    'attributes': {
        'color': {'value': 'Red', 'confidence': 'high'},
        'size': {'value': 'big', 'confidence': 'high'},
    },
}


def _labeler(content: Any, pack: Any) -> VlmLabeler:
    fake = MagicMock(spec=httpx.AsyncClient)
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value={'choices': [{'message': {'content': json.dumps(content)}}]})
    resp.raise_for_status = MagicMock()
    fake.post = AsyncMock(return_value=resp)
    return VlmLabeler(client=fake, pack=pack)


def _sent_text(labeler: VlmLabeler) -> str:
    content = labeler._client.post.await_args.kwargs['json']['messages'][1]['content']
    return ' '.join(p['text'] for p in content if p['type'] == 'text')


@pytest.mark.asyncio
async def test_a_classifying_call_asks_for_and_parses_the_declared_attributes() -> None:
    labeler = _labeler(REPLY, PACK)
    reply = await labeler.label_combined(
        'c1', _make_jpeg(), class_names=['car'], region_bboxes_norm=[(0.1, 0.1, 0.5, 0.5)]
    )
    assert 'color' in _sent_text(labeler)
    assert reply.attributes == {'color': AttributeValue(value='red', confidence='high')}


@pytest.mark.asyncio
async def test_a_call_that_does_not_classify_asks_for_no_attributes() -> None:
    labeler = _labeler(REPLY, PACK)
    reply = await labeler.label_combined(
        'c1', _make_jpeg(), class_names=None, region_bboxes_norm=[(0.1, 0.1, 0.5, 0.5)]
    )
    assert 'color' not in _sent_text(labeler)
    assert reply.attributes == {}


@pytest.mark.asyncio
async def test_a_pack_without_attributes_keeps_the_prompt_and_reply_as_before() -> None:
    labeler = _labeler(REPLY, GENERIC_REGION_PACK)
    reply = await labeler.label_combined(
        'c1', _make_jpeg(), class_names=['car'], region_bboxes_norm=[(0.1, 0.1, 0.5, 0.5)]
    )
    assert 'color' not in _sent_text(labeler)
    assert reply.attributes == {}


@pytest.mark.asyncio
async def test_a_batched_call_asks_for_and_parses_the_attributes_per_crop() -> None:
    labeler = _labeler({'results': [{'img': 1, **REPLY}, {'img': 2, **REPLY}]}, PACK)
    crops = [
        CombinedCrop(
            crop_id=cid,
            jpeg_bytes=_make_jpeg(),
            region_bboxes_norm=[(0.1, 0.1, 0.5, 0.5)],
            classify=True,
        )
        for cid in ('c1', 'c2')
    ]
    replies = await labeler.label_combined_batch(crops, class_names=['car'])
    assert 'color' in _sent_text(labeler)
    assert all(r is not None and r.attributes['color'].value == 'red' for r in replies.values())


META = RunMeta('rw_1', '2026-10-15T00:00:00+00:00', 'env@None', 'm', 'cars', 3)


def _update(reply: VlmCombinedReply, meta: RunMeta | None) -> dict[str, Any]:
    return _combined_class_update(
        reply, ['car'], vlm_model='m', name_to_id={'car': 4}, slot_meta=meta
    )


def _reply(**over: Any) -> VlmCombinedReply:
    return VlmCombinedReply(
        img_id='c1',
        class_id=0,
        class_confidence='high',
        region_visible=True,
        **{'attributes': {'color': AttributeValue(value='red', confidence='high')}, **over},
    )


def test_the_class_update_carries_the_assist_slot_when_attributes_were_answered() -> None:
    slot = _update(_reply(), META)['assist']
    assert slot['attributes'] == [{'key': 'color', 'value': 'red', 'confidence': 'high'}]
    assert (slot['class_name'], slot['run_id'], slot['prompt_pack_revision']) == ('car', 'rw_1', 3)


def test_no_attributes_or_no_meta_writes_no_slot() -> None:
    assert 'assist' not in _update(_reply(attributes={}), META)
    assert 'assist' not in _update(_reply(), None)


def test_the_slot_meta_names_the_runtimes_endpoint_and_pack() -> None:
    rt = SimpleNamespace(vlm_identity=VlmIdentity('env@None', 'm'), pack_ref=('cars', 3))
    meta = combined_slot_meta(rt)
    assert (meta.vlm_endpoint, meta.vlm_model, meta.prompt_pack, meta.prompt_pack_revision) == (
        'env@None',
        'm',
        'cars',
        3,
    )
    assert meta.run_id.startswith('rw_')
