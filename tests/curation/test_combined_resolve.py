"""W5: one combined VLM reply -> boxes/status/write fields, shared by the
worker's combined stage and the test-on-crop preview."""

from __future__ import annotations

import pytest

from scripts.curation.worker.combined_resolve import (
    CombinedResolution,
    resolve_combined_reply,
    should_classify,
)
from scripts.curation.worker.verify import TaskBoxInput
from src.config import DetectionProfile
from src.config.region_state import RegionStatus
from src.services.detection.region_text_rules import region_text_rules
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_labeler import VlmCombinedReply


PROFILE = DetectionProfile(name='wheels', text_reader='none')


def _cand(i: int, *, box_id: str | None = None, x: float = 0.1) -> TaskBoxInput:
    return TaskBoxInput(
        bbox_in_crop=(x, 0.5, x + 0.2, 0.9),
        bbox_in_source=(x / 2, 0.5, (x + 0.2) / 2, 0.9),
        score=0.9,
        detector='sam3',
        detector_version='1',
        source='segmenter',
        box_id=box_id,
    )


def _reply(*verdicts: bool | None, visible: bool = True) -> VlmCombinedReply:
    return VlmCombinedReply(
        img_id='c1',
        region_visible=visible,
        region_boxes=[
            VlmBoxVerdict(box=i + 1, bbox_correct=v, confidence='high' if v is not None else None)
            for i, v in enumerate(verdicts)
        ],
    )


async def _resolve(
    candidates: list[TaskBoxInput],
    reply: VlmCombinedReply | None,
    *,
    reverify: bool = False,
    force_resolve: bool = False,
) -> CombinedResolution:
    return await resolve_combined_reply(
        candidates,
        reply,
        item_bbox_norm=(0.0, 0.0, 1.0, 1.0),
        reverify=reverify,
        effective_class_names=None,
        name_to_id={},
        vlm_model='m',
        profile=PROFILE,
        rules=region_text_rules(PROFILE),
        ocr=None,
        crop_jpeg=None,
        crop_id='c1',
        force_resolve=force_resolve,
    )


@pytest.mark.asyncio
async def test_boxes_the_vlm_confirmed_are_accepted_and_the_wrong_one_is_rejected() -> None:
    res = await _resolve([_cand(0), _cand(1, x=0.5), _cand(2, x=0.7)], _reply(True, True, False))

    assert res.outcome == 'resolved'
    assert [b.state for b in res.boxes] == ['accepted', 'accepted', 'rejected']
    assert res.boxes[2].rejection_reason == 'region_visible_elsewhere'
    assert res.status == RegionStatus.DETECTED
    assert res.bbox_wrong is True
    assert res.trace == ['sam3:combined_verify_ok']
    assert res.extra['region_verified'] is True
    assert all(b.text is None for b in res.boxes), 'a text-free profile stores no text'


@pytest.mark.asyncio
async def test_no_verdict_on_any_box_is_a_retry_not_a_write_until_forced() -> None:
    res = await _resolve([_cand(0), _cand(1, x=0.5)], _reply(None, None))
    assert res.outcome == 'no_verdict'
    assert res.boxes == []

    forced = await _resolve([_cand(0), _cand(1, x=0.5)], _reply(None, None), force_resolve=True)
    assert forced.outcome == 'resolved'
    assert [b.rejection_reason for b in forced.boxes] == ['verifier_no_verdict'] * 2
    assert forced.status == RegionStatus.VERIFY_REJECTED
    assert forced.extra['region_verified'] is False


@pytest.mark.asyncio
async def test_a_missing_reply_is_a_no_verdict_too() -> None:
    assert (await _resolve([_cand(0)], None)).outcome == 'no_verdict'


@pytest.mark.asyncio
async def test_region_not_visible_on_a_fresh_pass_writes_no_box() -> None:
    res = await _resolve([_cand(0), _cand(1, x=0.5)], _reply(True, True, visible=False))

    assert res.no_region_visible is True
    assert res.boxes == []
    assert res.status == RegionStatus.NO_REGION_VISIBLE
    assert res.trace == ['sam3:combined_no_region_visible']


@pytest.mark.asyncio
async def test_region_not_visible_on_a_reverify_pass_rejects_the_stored_boxes_by_id() -> None:
    res = await _resolve(
        [_cand(0, box_id='b4'), _cand(1, box_id='b5', x=0.5)],
        _reply(True, True, visible=False),
        reverify=True,
    )

    assert [(b.box_id, b.state) for b in res.boxes] == [('b4', 'rejected'), ('b5', 'rejected')]
    assert res.status == RegionStatus.VERIFY_REJECTED


def _classify(
    *,
    class_validated: bool = False,
    class_source: str = 'vlm',
    test_holdout: bool = False,
    registry_loaded: bool = True,
) -> bool:
    return should_classify(
        class_validated=class_validated,
        class_source=class_source,
        test_holdout=test_holdout,
        class_confidence=0.0,
        registry_loaded=registry_loaded,
    )


def test_should_classify_follows_the_trust_rules() -> None:
    assert _classify() is True
    assert _classify(registry_loaded=False) is False
    assert _classify(class_validated=True) is False
    assert _classify(class_source='human') is False
    assert _classify(test_holdout=True) is False
