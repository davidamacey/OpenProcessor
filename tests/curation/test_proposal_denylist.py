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
