"""Labeler parse-failure logs carry lengths and codes, never reply text (#468)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from structlog.testing import capture_logs

from src.config import get_region_fields
from src.services.labeling.vlm_labeler import VlmLabeler
from src.services.labeling.vlm_models import CombinedCrop, RegionCrop


CANARY = 'ZZCANARY9Q'


def _assert_clean(logs: list[Any]) -> None:
    assert logs, 'expected the failure path to log'
    assert CANARY not in json.dumps(logs, default=str)


def test_verify_batch_unparseable_reply_not_logged() -> None:
    crops = [RegionCrop(crop_id='c1', jpeg_bytes=b'')]
    with capture_logs() as logs:
        VlmLabeler._parse_region_batch_response(f'{CANARY} not json', crops)
    _assert_clean(logs)


def test_verify_batch_misaligned_reply_not_logged() -> None:
    crops = [RegionCrop(crop_id='c1', jpeg_bytes=b''), RegionCrop(crop_id='c2', jpeg_bytes=b'')]
    raw = json.dumps([{'img': 1, 'is_region': True, 'text': CANARY}] * 5)
    with capture_logs() as logs:
        VlmLabeler._parse_region_batch_response(raw, crops)
    _assert_clean(logs)


def test_visibility_unparseable_reply_not_logged() -> None:
    crops = [RegionCrop(crop_id='c1', jpeg_bytes=b'')]
    with capture_logs() as logs:
        VlmLabeler._parse_region_visible_response(f'{CANARY} not json', crops, get_region_fields())
    _assert_clean(logs)


def test_verify_single_unparseable_reply_not_logged() -> None:
    with capture_logs() as logs:
        VlmLabeler._parse_region_response(
            f'{CANARY} not json', RegionCrop(crop_id='c', jpeg_bytes=b'')
        )
    _assert_clean(logs)


@pytest.mark.parametrize('raw', [f'{CANARY} not json', f'[{{"text": "{CANARY}"'])
def test_combined_batch_unparseable_reply_not_logged(raw: str) -> None:
    chunk = [CombinedCrop(crop_id='c1', jpeg_bytes=b'', region_bboxes_norm=[], classify=False)]
    with capture_logs() as logs:
        VlmLabeler._parse_combined_batch_response(raw, chunk, get_region_fields())
    _assert_clean(logs)
