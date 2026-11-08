"""The verify-region batch job omits crops the VLM gave no
verdict for -- it must never emit a synthesized ``is_region=False``
reject for a crop that simply got no answer."""

from __future__ import annotations

from typing import Any

import pytest

from src.services.labeling.vlm_client import VlmIdentity
from src.services.labeling.vlm_models import VlmRegionVerdict


pytestmark = pytest.mark.usefixtures('vlm_env_active')


class _Labeler:
    identity = VlmIdentity('env@None', 'test-vlm')

    def __init__(self, verdicts: list[VlmRegionVerdict]) -> None:
        self._verdicts = verdicts

    async def verify_region_batch(self, _crops: list[Any]) -> list[VlmRegionVerdict]:
        return self._verdicts


def _items(*crop_ids: str) -> list[dict[str, Any]]:
    return [
        {'crop_id': c, 'key': f'cache/vlm_batch/b/{c}.jpg', 'candidate_text': None}
        for c in crop_ids
    ]


@pytest.fixture(autouse=True)
def _staged(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.curation import vlm_batch

    monkeypatch.setattr(vlm_batch, 'read_staged', lambda _key: b'\xff\xd8\xff\xd9')


@pytest.mark.asyncio
async def test_no_verdict_crop_is_omitted_not_a_reject() -> None:
    from src.services.curation.vlm_batch import run_verify_region_batch

    labeler = _Labeler(
        [VlmRegionVerdict(crop_id='a', is_region=True, confidence='high', reason='ok')]
    )

    out = await run_verify_region_batch(labeler, _items('a', 'b'))

    assert {r['crop_id'] for r in out['results']} == {'a'}
    assert not any(r['reason'] == 'no_response' for r in out['results'])


@pytest.mark.asyncio
async def test_all_no_verdict_returns_empty_results() -> None:
    from src.services.curation.vlm_batch import run_verify_region_batch

    assert await run_verify_region_batch(_Labeler([]), _items('a')) == {'results': []}


@pytest.mark.asyncio
async def test_candidate_text_is_echoed_and_results_follow_input_order() -> None:
    from src.services.curation.vlm_batch import run_verify_region_batch

    labeler = _Labeler(
        [
            VlmRegionVerdict(crop_id='b', is_region=False, confidence='low', reason='r'),
            VlmRegionVerdict(crop_id='a', is_region=True, confidence='high', reason='ok'),
        ]
    )
    items = _items('a', 'b')
    items[0]['candidate_text'] = 'TEXT'

    out = await run_verify_region_batch(labeler, items)

    assert [(r['crop_id'], r['candidate_text']) for r in out['results']] == [
        ('a', 'TEXT'),
        ('b', None),
    ]
