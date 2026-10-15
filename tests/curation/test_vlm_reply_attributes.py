"""Attribute replies: declared keys only, closed vocabularies enforced, and
the assist / attributes labeler calls built on the parser."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from curation.test_proposal_denylist import _labeler
from src.services.labeling.vlm_attributes import AttributeSpec, parse_attributes
from src.services.labeling.vlm_labeler_assist import PackAttributesError
from src.services.labeling.vlm_models import ItemCrop
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


SPECS = [
    AttributeSpec(key='make', description='the manufacturer'),
    AttributeSpec(key='body_style', values=['sedan', 'suv']),
]


def _pack(*specs: AttributeSpec) -> Any:
    return dataclasses.replace(GENERIC_ITEM_PACK, attributes=list(specs))


def test_declared_keys_only_normalised_with_default_confidence() -> None:
    got, invalid = parse_attributes(
        {
            'make': {'value': '  Acme ', 'confidence': 'HIGH'},
            'colour': {'value': 'red', 'confidence': 'high'},
            'body_style': 'SUV',
        },
        SPECS,
    )
    assert invalid == 0
    assert {k: (v.value, v.confidence) for k, v in got.items()} == {
        'make': ('acme', 'high'),
        'body_style': ('suv', 'low'),
    }


def test_out_of_vocabulary_values_are_dropped_and_counted() -> None:
    got, invalid = parse_attributes(
        {'body_style': {'value': 'truck', 'confidence': 'high'}, 'make': {'value': 'zenith'}}, SPECS
    )
    assert list(got) == ['make']
    assert invalid == 1


@pytest.mark.parametrize('raw', [None, [], 'x', {'make': {'value': ''}}, {'make': None}])
def test_unusable_attribute_blocks_yield_nothing(raw: Any) -> None:
    assert parse_attributes(raw, SPECS) == ({}, 0)


def test_value_is_capped_at_48_characters() -> None:
    got, _ = parse_attributes({'make': {'value': 'x' * 80}}, SPECS)
    assert len(got['make'].value) == 48


@pytest.mark.asyncio
async def test_assist_batch_returns_class_and_attributes() -> None:
    entries = [
        {
            'img': 1,
            'class': 'car',
            'confidence': 'high',
            'attributes': {'make': {'value': 'Borealis', 'confidence': 'medium'}},
        },
        {
            'img': 2,
            'class': 'car',
            'confidence': 'low',
            'attributes': {'body_style': {'value': 'van'}},
        },
    ]
    labeler = _labeler(entries, _pack(*SPECS))
    crops = [ItemCrop(img_id=f'c{i}', jpeg_bytes=b'j') for i in (1, 2)]

    preds = await labeler.assist_batch(crops, ['car', 'bus'])

    assert [p.class_name for p in preds] == ['car', 'car']
    assert preds[0].attributes['make'].value == 'borealis'
    assert preds[1].attributes == {}
    assert preds[1].invalid_attribute_values == 1
    sent = labeler._client.post.await_args.kwargs['json']
    text = sent['messages'][1]['content'][0]['text']
    assert '- make: the manufacturer' in text
    assert '- body_style (one of: sedan, suv)' in text
    assert 'car, bus' in text
    assert sent['messages'][0]['content'] == GENERIC_ITEM_PACK.assist_system


@pytest.mark.asyncio
async def test_attributes_batch_reports_attributes_only() -> None:
    entries = [{'img': 1, 'attributes': {'make': {'value': 'zenith', 'confidence': 'high'}}}]
    labeler = _labeler(entries, _pack(*SPECS))

    (pred,) = await labeler.attributes_batch([ItemCrop(img_id='c1', jpeg_bytes=b'j')])

    assert pred.failure is None
    assert pred.attributes['make'].value == 'zenith'
    sent = labeler._client.post.await_args.kwargs['json']
    assert sent['messages'][0]['content'] == GENERIC_ITEM_PACK.attributes_system


@pytest.mark.asyncio
async def test_attributes_batch_marks_a_missing_entry_unparseable() -> None:
    labeler = _labeler([{'img': 2, 'attributes': {}}], _pack(*SPECS))
    (pred,) = await labeler.attributes_batch([ItemCrop(img_id='c1', jpeg_bytes=b'j')])
    assert pred.failure == 'unparseable'


@pytest.mark.asyncio
async def test_box_attributes_use_only_box_specs() -> None:
    box_spec = AttributeSpec(key='rim_material', values=['alloy'], applies_to='box')
    labeler = _labeler(
        [{'img': 1, 'attributes': {'rim_material': {'value': 'alloy'}, 'make': {'value': 'x'}}}],
        _pack(*SPECS, box_spec),
    )
    (pred,) = await labeler.attributes_batch(
        [ItemCrop(img_id='b1', jpeg_bytes=b'j')], applies_to='box'
    )
    assert list(pred.attributes) == ['rim_material']


@pytest.mark.asyncio
async def test_a_pack_without_attributes_cannot_serve_the_calls() -> None:
    labeler = _labeler([], GENERIC_ITEM_PACK)
    crops = [ItemCrop(img_id='c1', jpeg_bytes=b'j')]
    with pytest.raises(PackAttributesError):
        await labeler.assist_batch(crops, ['car'])
    with pytest.raises(PackAttributesError):
        await labeler.attributes_batch(crops)
    labeler._client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_pack_with_blank_assist_prompts_is_refused() -> None:
    pack = dataclasses.replace(_pack(*SPECS), assist_system='')
    labeler = _labeler([], pack)
    with pytest.raises(PackAttributesError, match='assist_system'):
        await labeler.assist_batch([ItemCrop(img_id='c1', jpeg_bytes=b'j')], ['car'])
