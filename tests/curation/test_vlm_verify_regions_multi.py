"""``POST /vlm/verify_regions`` over an item with several boxes.

One region crop is sent per verifiable stored box; each reply is written
onto its own box and the item status is re-derived. A box a human owns is
never sent, and a verdict for a box that moved while the VLM call was in
flight is dropped.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_region_fields
from src.config.curation import base_curation_config
from src.config.region_rejection import REJECT_REASON_HUMAN, REJECT_REASON_VERIFIER
from src.services.curation.region_boxes import RegionBox, boxes_write_fields
from src.services.labeling.vlm_client import VlmIdentity
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, prompt_pack_stamp


ITEMS = base_curation_config().items_index
F = get_region_fields()

pytestmark = pytest.mark.usefixtures('vlm_env')


def _box(box_id: str, x: float, **over: Any) -> RegionBox:
    kwargs: dict[str, Any] = {
        'box_id': box_id,
        'bbox_norm': (x, 0.1, x + 0.1, 0.3),
        'state': 'proposed',
        'score': 0.8,
        'detector': 'det_model',
        'source': 'detector',
    }
    kwargs.update(over)
    return RegionBox(**kwargs)


def _item(boxes: list[RegionBox], status: str = 'pending_verification') -> dict[str, Any]:
    return {
        'crop_id': 'c1',
        'image_path': '/data/c1.jpg',
        F.status: status,
        **boxes_write_fields(boxes, current_src={}),
    }


class _Verdict:
    def __init__(self, is_region: bool, reason: str) -> None:
        self.is_region = is_region
        self.reason = reason
        self.confidence = 'high'


class _Labeler:
    """Answers per call, in call order; records the crops it was shown."""

    identity = VlmIdentity('env@None', 'test-vlm')
    _pack = GENERIC_ITEM_PACK

    def __init__(self, answers: list[_Verdict | None]) -> None:
        self._answers = list(answers)
        self.crops: list[Any] = []

    async def verify_region(self, crop: Any) -> _Verdict | None:
        self.crops.append(crop)
        return self._answers.pop(0)


def _setup(monkeypatch: pytest.MonkeyPatch, labeler: Any, seen_boxes: list[Any]) -> None:
    import src.routers.curation.vlm as vlm_mod

    async def _no_pack(_os: Any) -> None:
        return None

    def _thumb(_path: Any, bbox: Any, **_kw: Any) -> bytes:
        seen_boxes.append(tuple(bbox))
        return b'jpeg-bytes'

    monkeypatch.setattr(vlm_mod, '_default_pack_name', _no_pack)
    monkeypatch.setattr(vlm_mod, '_get_vlm_labeler', lambda *_a, **_k: labeler)
    monkeypatch.setattr(vlm_mod, 'resolve_crop_root', lambda _path: SimpleNamespace())
    monkeypatch.setattr(vlm_mod, 'resolve_safe_path', lambda path, _root: path)
    monkeypatch.setattr(vlm_mod.THUMBNAIL_CACHE, 'get_or_compute', _thumb)


async def _verify(fake: QueryFakeOpenSearch) -> dict[str, Any]:
    import src.routers.curation.vlm as vlm_mod
    from src.routers.curation.vlm import VlmVerifyRegionsRequest

    return await vlm_mod.vlm_verify_regions(
        VlmVerifyRegionsRequest(crop_ids=['c1']), fake, object()
    )


@pytest.mark.asyncio
async def test_each_box_is_verified_and_the_status_re_derived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boxes = [_box('b1', 0.1), _box('b2', 0.3), _box('b3', 0.5)]
    fake = QueryFakeOpenSearch({ITEMS: {'c1': _item(boxes)}})
    labeler = _Labeler(
        [_Verdict(True, 'ok'), _Verdict(False, 'not a region'), _Verdict(True, 'ok')]
    )
    seen: list[Any] = []
    _setup(monkeypatch, labeler, seen)

    resp = await _verify(fake)

    assert resp == {'verified': 3}
    assert seen == [b.bbox_norm for b in boxes]
    doc = fake.docs(ITEMS)['c1']
    by_id = {b['box_id']: b for b in doc[F.boxes]}
    assert [by_id[i]['state'] for i in ('b1', 'b2', 'b3')] == ['accepted', 'rejected', 'accepted']
    assert by_id['b2']['rejection_reason'] == REJECT_REASON_VERIFIER
    assert by_id['b2']['bbox_correct'] is False
    assert by_id['b1']['bbox_correct'] is True
    assert doc[F.status] == 'detected'
    assert doc[F.count] == 2
    assert doc[F.rejected_count] == 1
    assert doc[F.verified] is True
    assert doc[F.reason] == 'not a region'
    assert doc['vlm_endpoint'] == 'env@None'
    assert doc['vlm_model'] == 'test-vlm'
    assert doc['vlm_prompt_pack'] == prompt_pack_stamp(GENERIC_ITEM_PACK)


@pytest.mark.asyncio
async def test_every_box_rejected_makes_the_item_verify_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'c1': _item([_box('b1', 0.1), _box('b2', 0.3)])}})
    _setup(monkeypatch, _Labeler([_Verdict(False, 'no'), _Verdict(False, 'no')]), [])

    await _verify(fake)

    doc = fake.docs(ITEMS)['c1']
    assert doc[F.status] == 'verify_rejected'
    assert doc[F.verified] is False
    assert doc[F.rejection_reason] == REJECT_REASON_VERIFIER


@pytest.mark.asyncio
async def test_a_human_owned_or_rejected_box_is_never_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boxes = [
        _box('b1', 0.1, state='accepted', source='human', detector='human'),
        _box('b2', 0.3, state='rejected', rejection_reason=REJECT_REASON_HUMAN),
        _box('b3', 0.5),
    ]
    fake = QueryFakeOpenSearch({ITEMS: {'c1': _item(boxes, 'detected')}})
    seen: list[Any] = []
    _setup(monkeypatch, _Labeler([_Verdict(False, 'no')]), seen)

    resp = await _verify(fake)

    assert resp == {'verified': 1}
    assert seen == [boxes[2].bbox_norm]
    by_id = {b['box_id']: b for b in fake.docs(ITEMS)['c1'][F.boxes]}
    assert by_id['b1']['state'] == 'accepted'
    assert by_id['b2']['state'] == 'rejected'
    assert by_id['b3']['state'] == 'rejected'


@pytest.mark.asyncio
async def test_a_box_with_no_verdict_is_left_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'c1': _item([_box('b1', 0.1), _box('b2', 0.3)])}})
    _setup(monkeypatch, _Labeler([None, _Verdict(True, 'ok')]), [])

    resp = await _verify(fake)

    assert resp == {'verified': 1}
    by_id = {b['box_id']: b for b in fake.docs(ITEMS)['c1'][F.boxes]}
    assert by_id['b1']['state'] == 'proposed'
    assert by_id['b2']['state'] == 'accepted'


@pytest.mark.asyncio
async def test_a_verdict_for_a_box_that_moved_mid_flight_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'c1': _item([_box('b1', 0.1), _box('b2', 0.3)])}})

    class _Moving(_Labeler):
        async def verify_region(self, crop: Any) -> _Verdict | None:
            verdict = await super().verify_region(crop)
            if len(self.crops) == 1:
                # A (non-human) writer moves b1 while the VLM answers.
                doc = fake.docs(ITEMS)['c1']
                doc[F.boxes][0] = {**doc[F.boxes][0], 'bbox_norm': [0.6, 0.6, 0.7, 0.7]}
            return verdict

    _setup(monkeypatch, _Moving([_Verdict(True, 'ok'), _Verdict(True, 'ok')]), [])

    await _verify(fake)

    by_id = {b['box_id']: b for b in fake.docs(ITEMS)['c1'][F.boxes]}
    assert by_id['b1']['state'] == 'proposed'
    assert by_id['b1']['bbox_norm'] == [0.6, 0.6, 0.7, 0.7]
    assert by_id['b2']['state'] == 'accepted'
