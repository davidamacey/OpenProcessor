"""Region-verify batch transport + parsing for the VLM labeler.

Mirrors ``test_vlm_class_reply.py``'s fix for the same defect in
``_verify_region_chunk``: batched verify calls sent no
``response_format`` and had no reasoning-channel fallback, so a
reasoning-parser deployment could return the whole verdict in
``reasoning_content`` and leave ``content`` empty -- every crop in that
chunk then read as a manufactured reject (``is_region=False``) instead
of what actually happened: no answer at all.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.services.labeling.vlm_labeler import RegionCrop, VlmLabeler


def _labeler(message: dict[str, Any] | None = None, *, fail: bool = False) -> VlmLabeler:
    fake = MagicMock(spec=httpx.AsyncClient)
    if fail:
        fake.post = AsyncMock(side_effect=httpx.ConnectError('down'))
    else:
        resp = MagicMock()
        resp.status_code = 200
        resp.json = MagicMock(return_value={'choices': [{'message': message or {}}]})
        resp.raise_for_status = MagicMock()
        fake.post = AsyncMock(return_value=resp)
    return VlmLabeler(client=fake)


def _crops(n: int) -> list[RegionCrop]:
    return [RegionCrop(crop_id=f'c{i}', jpeg_bytes=b'jpeg') for i in range(1, n + 1)]


def _entries(n: int, *, is_region: bool = True) -> list[dict[str, Any]]:
    return [{'img': i, 'is_region': is_region, 'confidence': 'high'} for i in range(1, n + 1)]


class TestVerifyBatchTransport:
    @pytest.mark.asyncio
    async def test_requests_json_object_mode_with_results_envelope(self) -> None:
        labeler = _labeler({'content': json.dumps({'results': _entries(2)})})
        verdicts = await labeler.verify_region_batch(_crops(2))
        sent = labeler._client.post.await_args.kwargs['json']
        assert sent['response_format'] == {'type': 'json_object'}
        user_text = ' '.join(
            part['text'] for part in sent['messages'][1]['content'] if part['type'] == 'text'
        )
        assert '"results"' in user_text
        assert [v.is_region for v in verdicts] == [True, True]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('reasoning_key', ['reasoning_content', 'reasoning'])
    @pytest.mark.parametrize('content', [None, '', ']'])
    async def test_answer_in_reasoning_channel_is_read(
        self, reasoning_key: str, content: str | None
    ) -> None:
        answer = json.dumps(_entries(3))
        labeler = _labeler({'content': content, reasoning_key: f'Looking at it... {answer}'})
        verdicts = await labeler.verify_region_batch(_crops(3))
        assert [v.is_region for v in verdicts] == [True, True, True]

    @pytest.mark.asyncio
    async def test_empty_reply_yields_no_verdicts_not_rejects(self) -> None:
        labeler = _labeler({'content': ''})
        verdicts = await labeler.verify_region_batch(_crops(2))
        assert verdicts == []

    @pytest.mark.asyncio
    async def test_unparseable_reply_yields_no_verdicts(self) -> None:
        labeler = _labeler({'content': 'definitely not json'})
        verdicts = await labeler.verify_region_batch(_crops(2))
        assert verdicts == []

    @pytest.mark.asyncio
    async def test_missing_entry_is_omitted_not_a_reject(self) -> None:
        labeler = _labeler({'content': json.dumps({'results': _entries(1)})})
        verdicts = await labeler.verify_region_batch(_crops(2))
        by_id = {v.crop_id: v for v in verdicts}
        assert by_id['c1'].is_region is True
        assert 'c2' not in by_id

    @pytest.mark.asyncio
    async def test_request_failure_yields_no_verdicts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        labeler = _labeler(fail=True)
        monkeypatch.setattr('src.services.labeling.vlm_client.RETRY_MAX_ATTEMPTS', 1)
        verdicts = await labeler.verify_region_batch(_crops(2))
        assert verdicts == []
