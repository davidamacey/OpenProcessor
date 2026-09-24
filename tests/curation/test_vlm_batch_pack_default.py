"""``POST /vlm/verify_region_batch`` and ``/vlm/region_visible_batch`` use the
``prompt_pack`` settings-doc default (like ``label_batch`` /
``verify_regions``), not only the ``OP_PROMPT_PACK_PATH`` pack."""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config.curation import CurationConfig
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, list[Any]]:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router, vlm

    paths = {}
    for name in ('pallet_v1', 'food_v2'):
        data = GENERIC_ITEM_PACK.to_dict()
        data['name'] = name
        paths[name] = tmp_path / f'{name}.json'
        paths[name].write_text(json.dumps(data))
    cfg = CurationConfig(prompt_pack_path=paths['pallet_v1'], prompt_pack_paths=(paths['food_v2'],))
    monkeypatch.setattr('src.config.curation.get_curation_config', lambda: cfg)
    monkeypatch.setattr(
        'src.clients.curation_opensearch.get_curation_settings',
        AsyncMock(return_value={'defaults': {'prompt_pack': 'food_v2'}}),
    )

    requested: list[Any] = []

    class _Labeler:
        async def verify_region_batch(self, _crops: list[Any]) -> list[Any]:
            return []

        async def region_visible_batch(self, crops: list[Any]) -> dict[str, bool]:
            return {c.crop_id: True for c in crops}

    def _fake_get(pack_name: str | None = None) -> _Labeler:
        requested.append(pack_name)
        return _Labeler()

    monkeypatch.setattr(vlm, '_get_vlm_labeler', _fake_get)
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: AsyncMock()
    return TestClient(app), requested


_IMG = base64.b64encode(b'\xff\xd8\xff\xd9').decode()


def test_region_visible_batch_uses_settings_default_pack(
    client: tuple[TestClient, list[Any]],
) -> None:
    c, requested = client
    r = c.post(
        '/curation/vlm/region_visible_batch',
        json={'items': [{'crop_id': 'a', 'image_b64': _IMG}]},
    )
    assert r.status_code == 200, r.text
    assert requested == ['food_v2']


def test_verify_region_batch_uses_settings_default_pack(
    client: tuple[TestClient, list[Any]],
) -> None:
    c, requested = client
    c.post(
        '/curation/vlm/verify_region_batch',
        json={'items': [{'crop_id': 'a', 'region_image_b64': _IMG}]},
    )
    assert requested == ['food_v2']
