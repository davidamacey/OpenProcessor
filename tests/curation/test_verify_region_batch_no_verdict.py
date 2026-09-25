"""``POST /curation/vlm/verify_region_batch`` omits crops the VLM gave no
verdict for -- it must never emit a synthesized ``is_region=False``
reject for a crop that simply got no answer."""

from __future__ import annotations

import base64
from typing import Any

import pytest

from src.services.labeling.vlm_labeler import VlmRegionVerdict


_IMG = base64.b64encode(b'\xff\xd8\xff\xd9').decode()


class _Labeler:
    def __init__(self, verdicts: list[VlmRegionVerdict]) -> None:
        self._verdicts = verdicts

    async def verify_region_batch(self, _crops: list[Any]) -> list[VlmRegionVerdict]:
        return self._verdicts


def _setup(monkeypatch: pytest.MonkeyPatch, labeler: _Labeler) -> None:
    import src.routers.curation.vlm as vlm_mod

    async def _no_pack(_os: Any) -> None:
        return None

    monkeypatch.setattr(vlm_mod, '_default_pack_name', _no_pack)
    monkeypatch.setattr(vlm_mod, '_get_vlm_labeler', lambda *_a, **_k: labeler)


@pytest.mark.asyncio
async def test_no_verdict_crop_is_omitted_not_a_reject(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.routers.curation.vlm as vlm_mod
    from src.routers.curation.vlm import VlmVerifyRegionBatchItem, VlmVerifyRegionBatchRequest

    _setup(
        monkeypatch,
        _Labeler([VlmRegionVerdict(crop_id='a', is_region=True, confidence='high', reason='ok')]),
    )

    resp = await vlm_mod.vlm_verify_region_batch(
        VlmVerifyRegionBatchRequest(
            items=[
                VlmVerifyRegionBatchItem(crop_id='a', region_image_b64=_IMG),
                VlmVerifyRegionBatchItem(crop_id='b', region_image_b64=_IMG),
            ]
        ),
        object(),  # opensearch is unused on this path
        object(),  # profile dependency, unused on this path
    )

    result_ids = {r.crop_id for r in resp.results}
    assert result_ids == {'a'}
    assert not any(r.reason == 'no_response' for r in resp.results)


@pytest.mark.asyncio
async def test_all_no_verdict_returns_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.routers.curation.vlm as vlm_mod
    from src.routers.curation.vlm import VlmVerifyRegionBatchItem, VlmVerifyRegionBatchRequest

    _setup(monkeypatch, _Labeler([]))

    resp = await vlm_mod.vlm_verify_region_batch(
        VlmVerifyRegionBatchRequest(
            items=[VlmVerifyRegionBatchItem(crop_id='a', region_image_b64=_IMG)]
        ),
        object(),
        object(),
    )

    assert resp.results == []
