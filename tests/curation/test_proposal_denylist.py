"""Per-pack proposal-name denylist: scene/quality words are not class proposals (#61)."""

from __future__ import annotations

import dataclasses
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.services.labeling.vlm_labeler import VlmLabeler
from src.services.labeling.vlm_models import ItemCrop
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, PromptPack, proposal_denied


def _labeler(entries: list[dict[str, Any]], pack: PromptPack = GENERIC_ITEM_PACK) -> VlmLabeler:
    fake = MagicMock(spec=httpx.AsyncClient)
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(
        return_value={'choices': [{'message': {'content': json.dumps({'results': entries})}}]}
    )
    resp.raise_for_status = MagicMock()
    fake.post = AsyncMock(return_value=resp)
    return VlmLabeler(client=fake, pack=pack)


def _entry(i: int, proposed: str) -> dict[str, Any]:
    return {'img': i, 'class': '__new__', 'confidence': 'high', 'proposed_class': proposed}


@pytest.mark.parametrize(
    ('slug', 'patterns', 'denied'),
    [
        ('blurry_object', ['blurry_*'], True),
        ('street_scene', ['*_scene'], True),
        ('Street_Scene', ['*_scene'], True),
        ('sports_car', ['blurry_*', '*_scene'], False),
        ('anything', [], False),
        ('blurry', ['blurry'], True),
    ],
)
def test_proposal_denied(slug: str, patterns: list[str], denied: bool) -> None:
    assert proposal_denied(slug, patterns) is denied


def test_default_pack_denies_issue_examples() -> None:
    assert proposal_denied('blurry_image', GENERIC_ITEM_PACK.proposal_denylist)
    assert proposal_denied('indoor_scene', GENERIC_ITEM_PACK.proposal_denylist)
    assert not proposal_denied('forklift', GENERIC_ITEM_PACK.proposal_denylist)


def test_pack_round_trips_denylist() -> None:
    pack = dataclasses.replace(GENERIC_ITEM_PACK, proposal_denylist=['x_*'])
    assert PromptPack.from_dict(pack.to_dict()).proposal_denylist == ['x_*']


@pytest.mark.asyncio
async def test_denied_proposal_is_dropped_and_real_one_kept() -> None:
    labeler = _labeler([_entry(1, 'blurry_thing'), _entry(2, 'forklift')])
    crops = [ItemCrop(img_id=f'c{i}', jpeg_bytes=b'j') for i in (1, 2)]
    preds = await labeler.label_or_propose_batch(crops, ['box'])
    assert preds[0].proposed_class == ''
    assert preds[1].proposed_class == 'forklift'


@pytest.mark.asyncio
async def test_empty_denylist_keeps_everything() -> None:
    pack = dataclasses.replace(GENERIC_ITEM_PACK, proposal_denylist=[])
    labeler = _labeler([_entry(1, 'blurry_thing')], pack)
    preds = await labeler.label_or_propose_batch([ItemCrop(img_id='c1', jpeg_bytes=b'j')], ['box'])
    assert preds[0].proposed_class == 'blurry_thing'


# The shipped default must catch the scene / quality words the live COCO oracle run
# produced (abstract_background was 60% of pending proposals) and nothing that is a
# real object class.
_COCO_NAMES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic_light',
    'fire_hydrant',
    'stop_sign',
    'parking_meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports_ball',
    'kite',
    'baseball_bat',
    'baseball_glove',
    'skateboard',
    'surfboard',
    'tennis_racket',
    'bottle',
    'wine_glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot_dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted_plant',
    'bed',
    'dining_table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell_phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy_bear',
    'hair_drier',
    'toothbrush',
    'forklift',
    'crane',
    'pallet',
]


@pytest.mark.parametrize(
    'noise',
    [
        'abstract_background',
        'abstract_object',
        'abstract',
        'abstract_pattern',
        'plain_background',
        'dark_background',
        'background',
        'unknown_object',
        'generic_object',
        'object',
        'blurry',
        'blurry_object',
        'blurred_image',
        'out_of_focus',
        'low_quality_image',
        'street_scene',
        'scene',
        'scene_unclear',
        'empty',
        'empty_frame',
        'unclear',
        'unidentified',
    ],
)
def test_default_pack_denies_scene_and_quality_words(noise: str) -> None:
    assert proposal_denied(noise, GENERIC_ITEM_PACK.proposal_denylist), noise


@pytest.mark.parametrize('name', _COCO_NAMES)
def test_default_pack_keeps_real_class_names(name: str) -> None:
    assert not proposal_denied(name, GENERIC_ITEM_PACK.proposal_denylist), name


def test_every_shipped_default_pack_inherits_the_denylist() -> None:
    from src.services.labeling import vlm_prompts

    for pack in (vlm_prompts.GENERIC_ITEM_PACK, vlm_prompts.GENERIC_REGION_PACK):
        assert proposal_denied('abstract_background', pack.proposal_denylist), pack.name
