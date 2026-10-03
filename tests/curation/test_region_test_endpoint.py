"""W5 (§5.2): ``POST /region_profiles/test`` runs a draft or saved profile's
legs over one stored crop and previews the item the worker would leave. It
writes nothing."""

from __future__ import annotations

import json
from typing import Any

import pytest

from curation.config_test_stack import ITEM_BOX, profile_body
from src.config import get_region_fields
from src.routers.curation._item_models import ItemDoc
from src.services.curation.region_boxes import RegionBox, boxes_write_fields
from src.services.curation.wire import region_box_to_wire, serialize_item


F = get_region_fields()
URL = '/region_profiles/test'

#: Crop-frame candidates as the segmenter serves them: the third is an NMS
#: duplicate of the first. The first two carry a mask polygon.
SEGMENTS: list[dict[str, Any]] = [
    {
        'bbox_norm': [0.1, 0.5, 0.3, 0.9],
        'score': 0.9,
        'mask_iou': 0.8,
        'mask_polygon': [[0.1, 0.5], [0.3, 0.5], [0.3, 0.9], [0.1, 0.9]],
    },
    {
        'bbox_norm': [0.5, 0.5, 0.7, 0.9],
        'score': 0.8,
        'mask_iou': 0.7,
        'mask_polygon': [[0.5, 0.5], [0.7, 0.5], [0.6, 0.9]],
    },
    {'bbox_norm': [0.11, 0.51, 0.31, 0.91], 'score': 0.7, 'mask_iou': None},
]


@pytest.fixture
def car(stack: Any) -> dict[str, Any]:
    stack.upstream.segments = SEGMENTS
    return stack.seed('car1', class_id=0, class_name='widget', class_source='item_model')


def _run(stack: Any, **body: Any) -> Any:
    return stack.post(URL, crop_id='car1', draft=profile_body(), **body)


def _only_leg(response: Any, leg: str) -> dict[str, Any]:
    return next(x for x in response.json()['legs'] if x['leg'] == leg)


def test_every_candidate_is_reported_with_what_the_selection_did_to_it(stack, car) -> None:
    response = _run(stack)

    assert response.status_code == 200, response.text
    body = response.json()
    assert [x['leg'] for x in body['legs']] == ['detector', 'segmenter']
    assert _only_leg(response, 'detector')['status'] == 'skipped'
    assert _only_leg(response, 'detector')['reason'] == 'no_detector_model'
    candidates = _only_leg(response, 'segmenter')['candidates']
    assert [c['candidate_index'] for c in candidates] == [0, 1, 2]
    assert [c['selected'] for c in candidates] == [True, True, False]
    assert [c['drop_reason'] for c in candidates] == [None, None, 'nms']
    assert all(c['state'] == 'proposed' and c['box_id'] is None for c in candidates)


def test_a_candidate_is_a_complete_box_wire_element(stack, car) -> None:
    stored = RegionBox(box_id='b1', bbox_norm=(0.2, 0.2, 0.4, 0.4), state='accepted')
    wire_keys = set(region_box_to_wire(car, stored, crop_id='car1'))

    candidates = _only_leg(_run(stack), 'segmenter')['candidates']

    for candidate in candidates:
        assert set(candidate) >= wire_keys
    from src.routers.curation._config_test_models import RegionTestCandidate

    assert set(RegionTestCandidate.model_fields) >= wire_keys


def test_candidate_boxes_use_the_stored_box_math_in_both_frames(stack, car) -> None:
    candidates = _only_leg(_run(stack), 'segmenter')['candidates']

    for candidate, raw in zip(candidates, SEGMENTS, strict=True):
        # The crop frame is what the segmenter answered in.
        assert candidate['bbox_in_parent'] == pytest.approx(raw['bbox_norm'], abs=1e-6)
        # The same box STORED on the item is served with the same crop-frame box.
        stored = {
            **car,
            **boxes_write_fields(
                [RegionBox(box_id='b1', bbox_norm=tuple(candidate['bbox_norm']), state='accepted')],
                current_src={},
            ),
        }
        served = serialize_item(stored, 'car1')['region_boxes'][0]
        assert served['bbox_in_parent'] == pytest.approx(candidate['bbox_in_parent'], abs=1e-6)
        px1, py1, px2, py2 = ITEM_BOX
        assert candidate['bbox_norm'][0] == pytest.approx(
            px1 + raw['bbox_norm'][0] * (px2 - px1), abs=1e-6
        )
        assert candidate['bbox_norm'][3] == pytest.approx(
            py1 + raw['bbox_norm'][3] * (py2 - py1), abs=1e-6
        )


def test_the_mask_polygon_is_served_in_both_frames_and_round_trips(stack, car) -> None:
    candidates = _only_leg(_run(stack), 'segmenter')['candidates']

    assert candidates[2]['mask_polygon'] is None
    px1, py1, px2, py2 = ITEM_BOX
    for candidate, raw in zip(candidates[:2], SEGMENTS[:2], strict=True):
        in_parent = candidate['mask_polygon_in_parent']
        assert in_parent == [list(p) for p in raw['mask_polygon']]
        for (sx, sy), (cx, cy) in zip(candidate['mask_polygon'], in_parent, strict=True):
            assert (sx - px1) / (px2 - px1) == pytest.approx(cx, abs=1e-6)
            assert (sy - py1) / (py2 - py1) == pytest.approx(cy, abs=1e-6)
        assert candidate['mask_iou'] == pytest.approx(raw['mask_iou'])


def test_the_segmenter_is_asked_for_masks_with_the_prompt_in_force(stack, car) -> None:
    _run(stack, segmenter_text_prompt='tyre')

    (sent,) = stack.upstream.segment_requests
    assert sent['text_prompt'] == 'tyre'
    assert sent['return_masks'] is True
    assert sent['max_candidates'] == 4


def test_the_preview_is_the_item_the_worker_would_leave(stack, car) -> None:
    response = _run(stack)

    preview = response.json()['preview_item']
    ItemDoc.model_validate(preview)
    assert response.json()['preview_basis'] == 'selection_accepted'
    assert preview['region_status'] == 'detected'
    assert [b['box_id'] for b in preview['region_boxes']] == ['b1', 'b2']
    assert preview['region_count'] == 2
    assert all(b['state'] == 'accepted' for b in preview['region_boxes'])
    assert all(
        v is None for b in preview['region_boxes'] for k, v in b.items() if k.startswith('text')
    ), 'a text-free profile stores no text on any box'
    assert preview['region_profile'] is None, 'a draft carries no profile stamp'
    assert preview['region_detector_chain'] == ['sam3:hit', 'sam3:accepted_unverified']
    # It is exactly serialize_item of the stored item with that write laid over it.
    write = {k: v for k, v in preview.items() if k.startswith('region_')}
    assert (
        write['region_boxes'][0]['bbox_norm']
        == _only_leg(response, 'segmenter')['candidates'][0]['bbox_norm']
    )


def test_a_test_run_writes_nothing_anywhere(stack, car) -> None:
    before = stack.snapshot()

    for body in (
        {'draft': profile_body()},
        {'draft': profile_body(), 'verify': True},
        {'draft': profile_body(max_regions_per_item=0)},
        {'profile_name': 'nope'},
    ):
        stack.upstream.vlm_content = '{}'
        stack.post(URL, crop_id='car1', **body)

    assert stack.snapshot() == before
    assert stack.items.write_calls == 0


def test_an_invalid_draft_is_a_422_with_the_report(stack, car) -> None:
    response = stack.post(URL, crop_id='car1', draft=profile_body(max_regions_per_item=0))

    assert response.status_code == 422, response.text
    detail = response.json()['detail']
    assert detail['error'] == 'profile_invalid'
    assert [e['code'] for e in detail['report']['errors']] == ['profile_field_range']
    assert stack.upstream.segment_requests == [], 'an invalid profile is refused before any call'


def test_a_missing_or_other_projects_crop_is_a_404(stack, car) -> None:
    response = stack.post(URL, crop_id='not-there', draft=profile_body())

    assert response.status_code == 404
    assert response.json()['detail']['error'] == 'crop_not_found'


def test_a_total_leg_failure_is_a_502_but_a_partial_one_is_a_200(stack, car) -> None:
    stack.upstream.segmenter_up = False

    down = _run(stack)

    assert down.status_code == 502, down.text
    assert down.json()['detail']['error'] == 'segmenter_error'


def test_the_item_class_gates_eligibility_like_the_worker(stack, car) -> None:
    stack.seed('dog1', class_id=1, class_name='gadget', class_source='item_model')

    eligible = stack.post(URL, crop_id='car1', draft=profile_body(parent_classes=['widget']))
    other = stack.post(URL, crop_id='dog1', draft=profile_body(parent_classes=['widget']))

    assert eligible.json()['item_eligible'] is True
    assert other.json()['item_eligible'] is False
    assert eligible.json()['testable'] is True
    assert eligible.json()['reason'] is None
    assert other.json()['testable'] is False
    assert 'widget' in other.json()['reason']


def test_eligibility_is_case_insensitive_like_the_worker(stack, car) -> None:
    stack.seed('dog2', class_id=1, class_name='Gadget', class_source='item_model')

    body = stack.post(URL, crop_id='dog2', draft=profile_body(parent_classes=['gadget'])).json()

    assert body['testable'] is True


def test_an_empty_segmenter_answer_previews_no_region_box(stack, car) -> None:
    stack.upstream.segments = []

    body = _run(stack).json()

    assert _only_leg_of(body, 'segmenter')['candidates'] == []
    assert body['preview_item']['region_status'] == 'no_region_box'
    assert body['preview_item']['region_boxes'] == []


def _only_leg_of(body: dict[str, Any], leg: str) -> dict[str, Any]:
    return next(x for x in body['legs'] if x['leg'] == leg)


def test_verify_runs_the_combined_call_and_the_verdicts_decide_the_boxes(stack, car) -> None:
    stack.upstream.vlm_content = json.dumps(
        {
            'class_id': 0,
            'class_confidence': 'high',
            'region_visible': True,
            'region_boxes': [
                {'box': 1, 'region_bbox_correct': True, 'region_confidence': 'high'},
                {'box': 2, 'region_bbox_correct': False, 'region_confidence': 'high'},
            ],
        }
    )

    response = _run(stack, verify=True)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body['preview_basis'] == 'vlm_verdicts'
    preview = body['preview_item']
    assert [(b['box_id'], b['state']) for b in preview['region_boxes']] == [
        ('b1', 'accepted'),
        ('b2', 'rejected'),
    ]
    assert (preview['region_count'], preview['region_rejected_count']) == (1, 1)
    assert preview['vlm_endpoint'] == body['verify']['vlm']['endpoint']
    assert preview['vlm_model'] == 'org/a-root'
    assert body['verify']['parse_ok'] is True
    # Two boxes were drawn on the crop the VLM was shown.
    (sent,) = stack.upstream.vlm_requests
    assert sent['_host'] == 'a.vlm.test'


class _DetectorPool:
    """A Triton pool whose ``wheel_det`` answers the given anchors
    (``[cx, cy, w, h, conf]`` in the 640 letterbox frame)."""

    def __init__(self, anchors: list[list[float]]) -> None:
        self.anchors = anchors
        self.calls: list[str] = []

    async def infer(
        self,
        model_name: str,
        inputs: list[Any],  # noqa: ARG002
        outputs: list[Any],  # noqa: ARG002
    ) -> Any:
        import numpy as np

        self.calls.append(model_name)
        raw = np.array([self.anchors], dtype=np.float32).transpose(0, 2, 1)

        class Result:
            @staticmethod
            def as_numpy(name: str) -> Any:
                assert name == 'output0'
                return raw

        return Result()


@pytest.fixture
def detector(stack: Any, car: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> Any:
    """``wheel_det`` is READY in Triton and in the profile."""
    import src.main as main_module
    from src.services.triton_control import TritonControlService

    async def repo_index(_self: Any) -> list[dict[str, Any]]:
        return [{'name': 'wheel_det', 'state': 'READY', 'version': '1'}]

    monkeypatch.setattr(TritonControlService, 'get_repository_index', repo_index)
    pool = _DetectorPool([[160, 320, 128, 128, 0.9], [480, 320, 128, 128, 0.8]])
    monkeypatch.setattr(main_module, 'get_async_triton_pool', lambda: pool)
    return pool


def test_a_detector_hit_is_what_the_worker_uses_and_the_segmenter_is_not_run(
    stack, detector
) -> None:
    response = stack.post(URL, crop_id='car1', draft=profile_body(detector_model='wheel_det'))

    assert response.status_code == 200, response.text
    body = response.json()
    detector_leg = _only_leg(response, 'detector')
    assert detector_leg['status'] == 'ok'
    assert [c['selected'] for c in detector_leg['candidates']] == [True, True]
    assert {c['source'] for c in detector_leg['candidates']} == {'detector'}
    assert {c['detector'] for c in detector_leg['candidates']} == {'wheel_det'}
    assert _only_leg(response, 'segmenter')['status'] == 'skipped'
    assert _only_leg(response, 'segmenter')['reason'] == 'detector_hit'
    assert stack.upstream.segment_requests == [], 'the worker never runs the segmenter after a hit'
    assert detector.calls == ['wheel_det']
    assert [b['source'] for b in body['preview_item']['region_boxes']] == ['detector', 'detector']
    assert body['preview_item']['region_detector_chain'] == [
        'wheel_det:hit',
        'wheel_det:accepted_unverified',
    ]


def test_a_detector_miss_falls_through_to_the_segmenter(stack, detector) -> None:
    detector.anchors = [[160, 320, 128, 128, 0.1]]  # under the profile's floor

    response = stack.post(URL, crop_id='car1', draft=profile_body(detector_model='wheel_det'))

    assert response.status_code == 200, response.text
    assert _only_leg(response, 'detector')['status'] == 'ok'
    assert _only_leg(response, 'detector')['candidates'] == []
    assert _only_leg(response, 'segmenter')['status'] == 'ok'
    assert len(stack.upstream.segment_requests) == 1
    chain = response.json()['preview_item']['region_detector_chain']
    assert chain[:2] == ['wheel_det:miss', 'sam3:hit']


@pytest.fixture
def unreachable_triton(detector: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.main as main_module

    def no_pool() -> Any:
        msg = 'triton pool unavailable'
        raise RuntimeError(msg)

    monkeypatch.setattr(main_module, 'get_async_triton_pool', no_pool)


def test_a_detector_failure_is_reported_on_its_leg_and_the_segmenter_still_runs(
    stack, unreachable_triton
) -> None:
    response = stack.post(URL, crop_id='car1', draft=profile_body(detector_model='wheel_det'))

    assert response.status_code == 200, response.text
    leg = _only_leg(response, 'detector')
    assert leg['status'] == 'error'
    assert 'triton pool unavailable' in leg['reason']
    assert _only_leg(response, 'segmenter')['status'] == 'ok'


def test_a_detector_failure_with_no_other_leg_is_a_502_detector_error(
    stack, unreachable_triton
) -> None:
    stack.upstream.segmenter_up = False

    response = stack.post(URL, crop_id='car1', draft=profile_body(detector_model='wheel_det'))

    assert response.status_code == 502, response.text
    assert response.json()['detail']['error'] == 'detector_error'
