"""W5 (§5.1): ``probe`` is the production call, observed.

For every call a test run can make, a canned upstream reply is parsed by
``probe`` and by the production labeler method, and the two parses are
equal; and the prompt / raw reply the probe reports are the ones the
upstream actually received and sent."""

from __future__ import annotations

import io
import json
from typing import Any

import httpx
import pytest
from PIL import Image

from src.services.labeling import vlm_client
from src.services.labeling.vlm_labeler import VlmLabeler
from src.services.labeling.vlm_models import CombinedCrop, ItemCrop, RegionCrop, VlmTransportError
from src.services.labeling.vlm_probe import ProbeCrop, probe


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (120, 80), (9, 9, 9)).save(buf, format='JPEG')
    return buf.getvalue()


class Upstream:
    """An OpenAI-shaped endpoint with one canned reply; records every request."""

    def __init__(self, content: str, *, reasoning: str | None = None) -> None:
        self.content = content
        self.reasoning = reasoning
        self.requests: list[dict[str, Any]] = []

    def labeler(self) -> VlmLabeler:
        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content))
            message: dict[str, Any] = {'content': self.content}
            if self.reasoning is not None:
                message['reasoning_content'] = self.reasoning
            return httpx.Response(200, json={'choices': [{'message': message}]})

        return VlmLabeler(
            base_url='http://vlm.test/v1',
            model='m',
            client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        )


def _combined_reply(n_boxes: int = 2) -> str:
    return json.dumps(
        {
            'class_id': 1,
            'class_confidence': 'high',
            'region_visible': True,
            'region_boxes': [
                {'box': i + 1, 'region_bbox_correct': i == 0, 'region_confidence': 'high'}
                for i in range(n_boxes)
            ],
        }
    )


BOXES = [(0.1, 0.1, 0.4, 0.4), (0.5, 0.5, 0.9, 0.9)]


async def _both(call: str, content: str, crops: list[ProbeCrop], *, classes: list[str] | None):
    """``(probe result, production result, upstream seen by the probe)``."""
    probed = Upstream(content)
    result = await probe(probed.labeler(), call, crops, class_names=classes)  # type: ignore[arg-type]
    produced = Upstream(content)
    labeler = produced.labeler()
    if call == 'combined' and len(crops) == 1:
        c = crops[0]
        production: Any = await labeler.label_combined(
            c.crop_id, c.jpeg, class_names=classes, region_bboxes_norm=c.region_boxes
        )
    elif call == 'combined':
        production = await labeler.label_combined_batch(
            [
                CombinedCrop(
                    crop_id=c.crop_id,
                    jpeg_bytes=c.jpeg,
                    region_bboxes_norm=c.region_boxes,
                    classify=bool(classes),
                )
                for c in crops
            ],
            class_names=classes,
        )
    elif call == 'classify':
        production = await labeler.label_item_batch(
            [ItemCrop(img_id=c.crop_id, jpeg_bytes=c.jpeg) for c in crops], classes or []
        )
    elif call == 'open_classify':
        production = await labeler.label_or_propose_batch(
            [ItemCrop(img_id=c.crop_id, jpeg_bytes=c.jpeg) for c in crops], classes or []
        )
    elif call == 'region_verify' and len(crops) == 1:
        production = await labeler.verify_region(
            RegionCrop(crop_id=crops[0].crop_id, jpeg_bytes=crops[0].jpeg)
        )
    elif call == 'region_verify':
        production = await labeler.verify_region_batch(
            [RegionCrop(crop_id=c.crop_id, jpeg_bytes=c.jpeg) for c in crops]
        )
    else:
        production = await labeler.region_visible_batch(
            [RegionCrop(crop_id=c.crop_id, jpeg_bytes=c.jpeg) for c in crops]
        )
    assert len(produced.requests) == 1
    return result, production, probed


CASES: list[tuple[str, str, int, list[str] | None]] = [
    ('combined', _combined_reply(), 1, ['widget', 'gadget']),
    (
        'combined',
        json.dumps(
            [
                {'img': 1, **json.loads(_combined_reply())},
                {'img': 2, **json.loads(_combined_reply())},
            ]
        ),
        2,
        ['widget', 'gadget'],
    ),
    (
        'classify',
        json.dumps({'results': [{'img': 1, 'class': 'widget', 'confidence': 'high'}]}),
        1,
        ['widget', 'gadget'],
    ),
    (
        'open_classify',
        json.dumps(
            {
                'results': [
                    {'img': 1, 'class': '__new__', 'confidence': 'low', 'proposed_class': 'gizmo'}
                ]
            }
        ),
        1,
        ['widget'],
    ),
    (
        'region_verify',
        json.dumps({'is_region': True, 'confidence': 'high', 'reason': 'a wheel'}),
        1,
        None,
    ),
    (
        'region_verify',
        json.dumps(
            {
                'results': [
                    {'img': 1, 'is_region': True, 'confidence': 'high', 'reason': 'x'},
                    {'img': 2, 'is_region': False, 'confidence': 'low', 'reason': 'y'},
                ]
            }
        ),
        2,
        None,
    ),
    (
        'region_visible',
        json.dumps({'results': [{'img': 1, 'region_visible': False}]}),
        1,
        None,
    ),
]


@pytest.mark.parametrize(('call', 'content', 'n', 'classes'), CASES)
@pytest.mark.asyncio
async def test_the_probe_parse_equals_the_production_parse_and_the_prompt_is_what_was_sent(
    call: str, content: str, n: int, classes: list[str] | None
) -> None:
    crops = [
        ProbeCrop(crop_id=f'c{i}', jpeg=_jpeg(), region_boxes=BOXES if call == 'combined' else [])
        for i in range(1, n + 1)
    ]

    result, production, upstream = await _both(call, content, crops, classes=classes)

    assert result.parse_ok, result.parse_error
    values = [e.value for e in result.parsed]
    if isinstance(production, dict):
        assert values == [production[c.crop_id] for c in crops]
    elif isinstance(production, list):
        assert values == production
    else:
        assert values == [production]
    # The reported prompt is the payload the upstream received, byte for byte.
    sent = upstream.requests[0]
    system = next(m for m in sent['messages'] if m['role'] == 'system')['content']
    user = next(m for m in sent['messages'] if m['role'] == 'user')['content']
    user_text = (
        user if isinstance(user, str) else next(p['text'] for p in user if p['type'] == 'text')
    )
    assert result.prompt_system == system
    assert result.prompt_user_text == user_text
    assert result.raw_reply == content
    assert result.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_a_reply_that_does_not_parse_is_a_result_not_an_error() -> None:
    upstream = Upstream('this is not json at all')
    result = await probe(
        upstream.labeler(),
        'combined',
        [ProbeCrop(crop_id='c1', jpeg=_jpeg(), region_boxes=BOXES)],
        class_names=['widget'],
    )

    assert result.parse_ok is False
    assert result.parse_error
    assert result.parsed[0].value is None
    assert result.raw_reply == 'this is not json at all'


@pytest.mark.asyncio
async def test_a_reasoning_channel_is_reported_alongside_the_reply() -> None:
    upstream = Upstream(_combined_reply(), reasoning='thinking out loud')
    result = await probe(
        upstream.labeler(),
        'combined',
        [ProbeCrop(crop_id='c1', jpeg=_jpeg(), region_boxes=BOXES)],
        class_names=['widget', 'gadget'],
    )

    assert result.reasoning == 'thinking out loud'


@pytest.mark.parametrize('call', ['combined', 'classify', 'region_verify', 'region_visible'])
@pytest.mark.asyncio
async def test_an_upstream_failure_is_a_transport_error_for_every_call(
    call: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An outage must never read as "the model answered with nothing"."""
    monkeypatch.setattr(vlm_client, 'RETRY_WAIT_MIN_S', 0.0)

    def down(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('refused')

    labeler = VlmLabeler(
        base_url='http://vlm.test/v1',
        model='m',
        client=httpx.AsyncClient(transport=httpx.MockTransport(down)),
    )
    crops = [ProbeCrop(crop_id=f'c{i}', jpeg=_jpeg(), region_boxes=BOXES) for i in (1, 2)]
    for subset in (crops[:1], crops):
        with pytest.raises(VlmTransportError):
            await probe(labeler, call, subset, class_names=['widget'])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_capture_is_scoped_to_the_probe() -> None:
    """Ordinary labeler calls outside a probe record nothing anywhere."""
    upstream = Upstream(_combined_reply())
    labeler = upstream.labeler()
    await labeler.label_combined('c1', _jpeg(), class_names=['a', 'b'], region_bboxes_norm=BOXES)

    assert vlm_client._CHAT_CAPTURE.get() is None
