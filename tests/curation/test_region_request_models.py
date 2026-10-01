"""The whole-set region request bodies take ``region_label_source`` and
nothing spelled ``label_source`` (``extra='forbid'`` makes a stale key a 422)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.routers.curation._common_models import CropBatchStatusRequest, ItemRegionMetaRequest


def test_region_meta_takes_region_label_source() -> None:
    body = ItemRegionMetaRequest.model_validate(
        {'region_status': 'false_positive', 'region_label_source': 'human'}
    )
    assert body.region_label_source == 'human'


@pytest.mark.parametrize(
    'payload',
    [
        {'region_status': 'false_positive', 'label_source': 'human'},
        {'label_source': 'human'},
    ],
)
def test_region_meta_rejects_the_item_level_label_source_key(payload: dict) -> None:
    with pytest.raises(ValidationError):
        ItemRegionMetaRequest.model_validate(payload)


def test_batch_status_takes_region_label_source_and_rejects_label_source() -> None:
    ok = CropBatchStatusRequest.model_validate(
        {'crop_ids': ['a'], 'region_status': 'detected', 'region_label_source': 'human'}
    )
    assert ok.region_label_source == 'human'
    with pytest.raises(ValidationError):
        CropBatchStatusRequest.model_validate(
            {'crop_ids': ['a'], 'region_status': 'detected', 'label_source': 'human'}
        )
