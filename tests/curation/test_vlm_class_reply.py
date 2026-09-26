"""Class-answer transport + parsing for the VLM labeler.

Live evidence (2026-09-24): against a vLLM server running a reasoning
parser, the open-vocabulary class call (no JSON-object mode) came back
with an empty ``content`` or a lone ``"]"`` for most chunks — the answer
went to the reasoning channel — and every crop in those chunks was
recorded as an empty class answer. The combined call, which requests
JSON-object mode, parsed fine. These tests pin the class calls to the
same transport contract and make the parsers read what the VLM actually
answered instead of discarding it.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.config import get_region_fields
from src.services.labeling.vlm_labeler import (
    CombinedCrop,
    ItemCrop,
    VlmLabeler,
    _combined_reply_from_entry,
)


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


def _crops(n: int) -> list[ItemCrop]:
    return [ItemCrop(img_id=f'c{i}', jpeg_bytes=b'jpeg') for i in range(1, n + 1)]


def _entries(n: int, cls: str = 'widget') -> list[dict[str, Any]]:
    return [{'img': i, 'class': cls, 'confidence': 'high'} for i in range(1, n + 1)]


class TestClassCallTransport:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('open_vocab', [True, False])
    async def test_requests_json_object_mode_with_results_envelope(self, open_vocab: bool) -> None:
        labeler = _labeler({'content': json.dumps({'results': _entries(2)})})
        if open_vocab:
            preds = await labeler.label_or_propose_batch(_crops(2), ['widget'])
        else:
            preds = await labeler.label_item_batch(_crops(2), ['widget'])
        sent = labeler._client.post.await_args.kwargs['json']
        assert sent['response_format'] == {'type': 'json_object'}
        user_text = ' '.join(
            part['text'] for part in sent['messages'][1]['content'] if part['type'] == 'text'
        )
        assert '"results"' in user_text
        assert [p.class_name for p in preds] == ['widget', 'widget']

    @pytest.mark.asyncio
    async def test_catalog_prompt_asks_for_results_envelope(self) -> None:
        labeler = _labeler({'content': json.dumps({'results': _entries(1)})})
        await labeler.label_or_propose_batch(_crops(1), ['widget'], class_catalog='g: widget')
        sent = labeler._client.post.await_args.kwargs['json']
        assert sent['response_format'] == {'type': 'json_object'}
        assert '"results"' in sent['messages'][1]['content'][0]['text']

    @pytest.mark.asyncio
    @pytest.mark.parametrize('reasoning_key', ['reasoning_content', 'reasoning'])
    @pytest.mark.parametrize('content', [None, '', ']'])
    async def test_answer_in_reasoning_channel_is_read(
        self, reasoning_key: str, content: str | None
    ) -> None:
        answer = json.dumps(_entries(3, 'gadget'))
        labeler = _labeler({'content': content, reasoning_key: f'Looking at it... {answer}'})
        preds = await labeler.label_or_propose_batch(_crops(3), ['gadget'])
        assert [p.class_name for p in preds] == ['gadget'] * 3
        assert all(p.failure is None for p in preds)

    @pytest.mark.asyncio
    async def test_single_object_reply_for_single_crop(self) -> None:
        labeler = _labeler({'content': json.dumps({'img': 1, 'class': 'widget'})})
        preds = await labeler.label_or_propose_batch(_crops(1), ['widget'])
        assert preds[0].class_name == 'widget'

    @pytest.mark.asyncio
    async def test_unparseable_reply_is_flagged(self) -> None:
        labeler = _labeler({'content': ']'})
        preds = await labeler.label_or_propose_batch(_crops(2), ['widget'])
        assert [p.class_name for p in preds] == ['', '']
        assert [p.failure for p in preds] == ['unparseable', 'unparseable']

    @pytest.mark.asyncio
    async def test_missing_entry_is_flagged_unparseable(self) -> None:
        labeler = _labeler({'content': json.dumps({'results': _entries(1)})})
        preds = await labeler.label_or_propose_batch(_crops(2), ['widget'])
        assert preds[0].failure is None
        assert (preds[1].class_name, preds[1].failure) == ('', 'unparseable')

    @pytest.mark.asyncio
    async def test_empty_class_in_parsed_entry_is_not_a_failure(self) -> None:
        labeler = _labeler({'content': json.dumps({'results': _entries(1, '')})})
        preds = await labeler.label_or_propose_batch(_crops(1), ['widget'])
        assert (preds[0].class_name, preds[0].failure) == ('', None)

    @pytest.mark.asyncio
    async def test_request_failure_is_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        labeler = _labeler(fail=True)
        monkeypatch.setattr('src.services.labeling.vlm_client.RETRY_MAX_ATTEMPTS', 1)
        preds = await labeler.label_or_propose_batch(_crops(2), ['widget'])
        assert [p.failure for p in preds] == ['request_failed', 'request_failed']


class TestCombinedClassAnswer:
    NAMES = ['widget', 'gadget', 'sprocket']

    def _reply(self, **entry: Any):
        return _combined_reply_from_entry(
            {'region_visible': True, **entry},
            img_id='c1',
            fields=get_region_fields(),
            class_names=self.NAMES,
        )

    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [(1, 1), (1.0, 1), ('2', 2), (' 2 ', 2), ('1=gadget', 1), ('-1', -1), (-1, -1)],
    )
    def test_index_forms(self, raw: Any, expected: int) -> None:
        reply = self._reply(class_id=raw)
        assert (reply.class_id, reply.class_raw) == (expected, '')

    @pytest.mark.parametrize('raw', ['gadget', 'Gadget', ' GADGET '])
    def test_registry_name_in_class_id_resolves_to_index(self, raw: str) -> None:
        reply = self._reply(class_id=raw)
        assert (reply.class_id, reply.class_raw) == (1, '')

    def test_unknown_name_is_kept_as_raw_answer(self) -> None:
        reply = self._reply(class_id='zeppelin')
        assert (reply.class_id, reply.class_raw) == (None, 'zeppelin')
        # The region half of the reply survives a non-index class answer.
        assert reply.region_visible is True

    def test_class_key_is_read_when_class_id_absent(self) -> None:
        assert self._reply(**{'class': 'sprocket'}).class_id == 2
        assert self._reply(class_name='zeppelin').class_raw == 'zeppelin'

    @pytest.mark.parametrize('raw', [None, True, '', '  ', 'null', 'none'])
    def test_no_answer(self, raw: Any) -> None:
        reply = self._reply(class_id=raw)
        assert (reply.class_id, reply.class_raw) == (None, '')

    def test_name_without_catalog_is_raw(self) -> None:
        reply = _combined_reply_from_entry(
            {'region_visible': False, 'class_id': 'gadget'},
            img_id='c1',
            fields=get_region_fields(),
            class_names=None,
        )
        assert (reply.class_id, reply.class_raw) == (None, 'gadget')


class TestCombinedNestedEntryUnwrap:
    """Live evidence (2026-09-24): a reasoning model sometimes nests the
    whole per-image answer one level down under an invented key instead
    of the flat shape the prompt asks for, e.g.
    ``{"img": 2, "layout_analysis": {"region_visible": true, ...}}``.
    That read as "no region_visible answer" -- a no-verdict, retried
    forever -- even though the VLM did answer, just under the wrong key.
    """

    def test_single_nested_key_is_unwrapped(self) -> None:
        entry = {
            'img': 2,
            'layout_analysis': {
                'class_id': None,
                'class_confidence': None,
                'region_visible': True,
                'region_bbox_correct': None,
                'region_text': '782CCB',
                'region_confidence': 'high',
                'make': None,
                'model': None,
            },
        }
        reply = _combined_reply_from_entry(
            entry, img_id='c1', fields=get_region_fields(), class_names=None
        )
        assert reply.region_visible is True
        assert reply.region_text_reply == '782CCB'

    def test_a_different_invented_key_name_is_also_unwrapped(self) -> None:
        entry = {
            'img': 6,
            'interim_results': {
                'class_id': 11,
                'class_confidence': 'high',
                'region_visible': True,
                'region_bbox_correct': True,
            },
        }
        reply = _combined_reply_from_entry(
            entry,
            img_id='c1',
            fields=get_region_fields(),
            class_names=[f'c{i}' for i in range(20)],
        )
        assert reply.region_visible is True
        assert reply.class_id == 11

    def test_two_nested_candidates_is_ambiguous_stays_no_verdict(self) -> None:
        entry = {
            'img': 3,
            'first_guess': {'region_visible': True},
            'second_guess': {'region_visible': False},
        }
        with pytest.raises(ValueError, match='region_visible'):
            _combined_reply_from_entry(
                entry, img_id='c1', fields=get_region_fields(), class_names=None
            )

    def test_batch_parse_unwraps_nested_entries(self) -> None:
        raw = json.dumps(
            {
                'results': [
                    {
                        'img': 1,
                        'layout_analysis': {
                            'region_visible': True,
                            'region_text': '782CCB',
                        },
                    },
                    {
                        'img': 2,
                        'interim_results': {
                            'class_id': 0,
                            'region_visible': True,
                            'region_bbox_correct': True,
                        },
                    },
                ]
            }
        )
        chunk = [
            CombinedCrop(crop_id='c1', jpeg_bytes=b'x'),
            CombinedCrop(crop_id='c2', jpeg_bytes=b'x'),
        ]
        out = VlmLabeler._parse_combined_batch_response(
            raw, chunk, get_region_fields(), class_names=['widget']
        )
        assert out['c1'] is not None
        assert out['c1'].region_text_reply == '782CCB'
        assert out['c2'] is not None
        assert out['c2'].class_id == 0


def test_batch_entry_failure_is_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.services.labeling.vlm_labeler as mod

    # No region_visible answer: the entry is rejected (fail-closed), and
    # the rejection is now visible in the logs instead of silent.
    raw = json.dumps({'results': [{'img': 1, 'class_id': 0}]})
    chunk = [CombinedCrop(crop_id='c1', jpeg_bytes=b'x')]
    logged: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(mod.logger, 'warning', lambda event, **kw: logged.append((event, kw)))
    out = VlmLabeler._parse_combined_batch_response(
        raw, chunk, get_region_fields(), class_names=['widget']
    )
    assert out == {'c1': None}
    events = [e for e, _ in logged]
    assert 'vlm_labeler.combined_batch_entry_invalid' in events
