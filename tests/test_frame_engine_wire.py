"""Frame engine protocol version 1: the plan is taken exactly as written or refused."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.domain import frame_engine_wire as wire
from src.domain.processing_plan import BRANCHES


DETECTOR = {'model': 'yolov11_small_trt_end2end', 'input_size': 640, 'letterbox_fill': 114}
EMBED = {'model': 'pe_image_encoder', 'size': 336, 'batch': 32}
OCR = {
    'detector': 'paddleocr_det_trt_uint8',
    'recogniser': 'paddleocr_rec_trt_uint8',
    'batch': 16,
    'max_lines': 128,
    'det_side': 960,
}


def _plan(**over: Any) -> str:
    body: dict[str, Any] = {
        'version': 1,
        'branches': ['detect', 'embed_frame', 'embed_crops'],
        'detector': DETECTOR,
        'embed': EMBED,
        'crops': 'inline',
    }
    body.update(over)
    return json.dumps({k: v for k, v in body.items() if v is not None})


def test_the_ingest_plan_parses_and_round_trips() -> None:
    plan = wire.parse_plan(_plan(), has_boxes=False)
    assert plan.branches == ('detect', 'embed_frame', 'embed_crops')
    assert plan.detector == wire.DetectorSpec('yolov11_small_trt_end2end', 640, 114)
    assert plan.embed == wire.EmbedSpec('pe_image_encoder', 336, 32, 'whole_frame', 'resize_crop')
    assert plan.crops == 'inline'
    assert plan.embed_classes is None
    assert wire.parse_plan(plan.to_json(), has_boxes=False) == plan


def test_the_engine_branches_are_the_plan_vocabulary_minus_api_branches() -> None:
    assert set(wire.ENGINE_BRANCHES) <= set(BRANCHES)
    assert set(BRANCHES) - set(wire.ENGINE_BRANCHES) == {'near_duplicates', 'region'}
    assert set(wire.ENGINE_BRANCHES) >= wire.BUILT_BRANCHES - {wire.DERIVATIVES}


@pytest.mark.parametrize(
    ('text', 'fault'),
    [
        ('{', 'bad_json'),
        ('[]', 'bad_field'),
        (_plan(version=2), 'bad_version'),
        (_plan(version=None), 'bad_version'),
        (_plan(extra=1), 'unknown_field'),
        (_plan(branches=[]), 'empty_plan'),
        (_plan(branches=['detect', 'region']), 'unknown_branch'),
        (_plan(branches=['detect', 'detect']), 'duplicate_branch'),
        (_plan(branches=['embed_crops'], crops='none'), 'missing_dependency'),
        (_plan(branches=['embed_frame'], crops='inline'), 'missing_dependency'),
        (_plan(branches=['face_embed']), 'missing_dependency'),
        (_plan(detector=None), 'missing_field'),
        (_plan(embed=None), 'missing_field'),
        (_plan(detector={**DETECTOR, 'input_size': 0}), 'bad_field'),
        (_plan(detector={**DETECTOR, 'letterbox_fill': 256}), 'bad_field'),
        (_plan(detector={**DETECTOR, 'nms': 1}), 'unknown_field'),
        (_plan(embed={**EMBED, 'batch': True}), 'bad_field'),
        (_plan(embed={**EMBED, 'crop_recipe': 'letterbox'}), 'bad_field'),
        (_plan(crops='thumb'), 'bad_field'),
        (_plan(embed_classes=['car']), 'bad_field'),
        (_plan(branches=['detect', 'embed_frame'], embed_classes=[2]), 'bad_field'),
        (_plan(if_detected={'embed_crops': [2]}), 'condition_on_unconditional'),
        (_plan(if_detected={'ocr': [2]}), 'condition_on_unconditional'),
        (_plan(branches=['ocr'], crops='none', if_detected={'ocr': [2]}), 'missing_field'),
        (_plan(branches=['detect', 'ocr'], ocr=OCR, if_detected={'ocr': [2]}), 'bad_field'),
        (_plan(derivatives={'thumb': 256}), 'bad_field'),
    ],
)
def test_a_bad_plan_is_refused_with_its_fault(text: str, fault: str) -> None:
    with pytest.raises(wire.EnginePlanError) as err:
        wire.parse_plan(text, has_boxes=False)
    assert err.value.fault == fault


def test_stored_boxes_stand_in_for_the_detector() -> None:
    plan = wire.parse_plan(
        _plan(branches=['embed_frame', 'embed_crops'], detector=None, crops='none'), has_boxes=True
    )
    assert plan.branches == ('embed_frame', 'embed_crops')
    with pytest.raises(wire.EnginePlanError) as err:
        wire.parse_plan(_plan(), has_boxes=True)
    assert err.value.fault == 'boxes_with_detect'
    with pytest.raises(wire.EnginePlanError) as err:
        wire.parse_plan(
            _plan(branches=['embed_frame'], detector=None, crops='none'), has_boxes=True
        )
    assert err.value.fault == 'unused_boxes'
    with pytest.raises(wire.EnginePlanError) as err:
        wire.parse_plan(
            _plan(branches=['embed_frame', 'embed_crops'], detector=None, crops='none',
                  embed_classes=[1]),
            has_boxes=True,
        )  # fmt: skip
    assert err.value.fault == 'bad_field'


def test_unbuilt_branches_are_reported_unsupported_and_renditions_are_built() -> None:
    plan = wire.parse_plan(
        _plan(
            branches=['detect', 'faces', 'face_embed'],
            embed=None,
            crops='none',
            if_detected={'faces': [0]},
            derivatives=[{'name': 'thumb', 'side': 256}],
        ),
        has_boxes=False,
    )
    assert plan.unsupported() == ('faces', 'face_embed')
    assert plan.reported() == ('detect', 'faces', 'face_embed', 'derivatives')
    assert plan.if_detected == {'faces': (0,)}


def test_branch_status_round_trips_one_entry_per_name() -> None:
    states: dict[str, tuple[wire.BranchState, str]] = {
        'detect': ('ok', ''),
        'embed_crops': ('error', wire.DEPENDENCY_FAILED),
        'ocr': ('not_run', ''),
    }
    text = wire.encode_branch_status(states)
    assert json.loads(text) == {
        'detect': {'status': 'ok'},
        'embed_crops': {'status': 'error', 'reason': 'dependency_failed'},
        'ocr': {'status': 'not_run'},
    }
    assert wire.decode_branch_status(text) == states
    with pytest.raises(ValueError, match='bad status'):
        wire.decode_branch_status('{"detect": {"status": "fine"}}')


def test_expected_outputs_follow_the_branches() -> None:
    full = wire.parse_plan(_plan(), has_boxes=False)
    assert set(wire.expected_outputs(full)) == {
        'det_boxes', 'det_scores', 'det_classes', 'letterbox', 'blur_sums',
        'whole_embedding', 'crop_embeddings', 'crop_embedded', 'crop_jpegs',
    }  # fmt: skip
    frame_only = wire.parse_plan(
        _plan(branches=['embed_frame'], detector=None, crops='none'), has_boxes=False
    )
    assert wire.expected_outputs(frame_only) == ('whole_embedding',)
    assert set(wire.OUTPUT_DATATYPES) >= set(wire.expected_outputs(full))
    assert wire.INPUT_DATATYPES['boxes'] == 'FP64'


# ------------------------------------------------------------------ the ocr branch


def test_the_ocr_spec_parses_with_the_reference_defaults_and_round_trips() -> None:
    plan = wire.parse_plan(
        _plan(branches=['ocr'], detector=None, embed=None, crops='none', ocr=OCR), has_boxes=False
    )
    o = plan.ocr
    assert o is not None
    assert (o.thresh, o.box_thresh, o.unclip_ratio) == (0.2, 0.45, 1.4)  # PP-OCRv6 small yml
    assert (o.det_limit, o.det_max_side, o.det_antialias, o.taps) == ('max', 1920, False, 1)
    assert plan.unsupported() == ()
    assert wire.parse_plan(plan.to_json(), has_boxes=False) == plan


def test_ocr_needs_its_spec_and_the_spec_needs_the_branch() -> None:
    with pytest.raises(wire.EnginePlanError) as err:
        wire.parse_plan(
            _plan(branches=['ocr'], detector=None, embed=None, crops='none'), has_boxes=False
        )
    assert (err.value.fault, err.value.detail) == ('missing_field', 'ocr')
    with pytest.raises(wire.EnginePlanError) as err:
        wire.parse_plan(_plan(ocr=OCR), has_boxes=False)
    assert (err.value.fault, err.value.detail) == ('bad_field', 'ocr')


@pytest.mark.parametrize(
    ('change', 'fault'),
    [
        ({'batch': 0}, 'bad_field'),
        ({'batch': 17}, 'bad_field'),  # the recogniser profile's maximum
        ({'det_max_side': 1921}, 'bad_field'),  # the detector profile's maximum
        ({'det_limit': 'long'}, 'bad_field'),
        ({'det_antialias': 1}, 'bad_field'),
        ({'thresh': 1.5}, 'bad_field'),
        ({'taps': 9}, 'bad_field'),
        ({'nms': 0.5}, 'unknown_field'),
    ],
)
def test_a_bad_ocr_spec_is_refused(change: dict[str, Any], fault: str) -> None:
    text = _plan(branches=['ocr'], detector=None, embed=None, crops='none', ocr={**OCR, **change})
    with pytest.raises(wire.EnginePlanError) as err:
        wire.parse_plan(text, has_boxes=False)
    assert err.value.fault == fault


def test_the_ocr_outputs_are_declared_with_their_datatypes() -> None:
    plan = wire.parse_plan(
        _plan(branches=['ocr'], detector=None, embed=None, crops='none', ocr=OCR), has_boxes=False
    )
    names = wire.expected_outputs(plan)
    assert names == (
        'ocr_quads',
        'ocr_texts',
        'ocr_scores',
        'ocr_char_min',
        'ocr_det_scores',
        'ocr_overflow',
    )
    assert [wire.OUTPUT_DATATYPES[n] for n in names] == [
        'FP32',
        'BYTES',
        'FP32',
        'FP32',
        'FP32',
        'INT32',
    ]


def test_renditions_are_parsed_with_defaults_and_round_trip() -> None:
    plan = wire.parse_plan(
        _plan(
            derivatives=[
                {'name': 'thumb', 'side': 256},
                {'name': 'grid', 'side': 1024, 'quality': 70},
            ]
        ),
        has_boxes=False,
    )
    assert [(r.name, r.side, r.quality, r.format) for r in plan.derivatives] == [
        ('thumb', 256, 85, 'jpeg'),
        ('grid', 1024, 70, 'jpeg'),
    ]
    assert wire.parse_plan(plan.to_json(), has_boxes=False) == plan
    assert 'derivative_jpegs' in wire.expected_outputs(plan)


@pytest.mark.parametrize(
    'entry',
    [
        {'name': 'thumb'},
        {'name': 'thumb', 'side': 8},
        {'name': 'thumb', 'side': 256, 'quality': 0},
        {'name': 'thumb', 'side': 256, 'format': 'webp'},
        {'name': 'thumb', 'side': 256, 'crop': True},
        {'name': '', 'side': 256},
    ],
)
def test_a_bad_rendition_refuses_the_plan(entry: dict) -> None:
    with pytest.raises(wire.EnginePlanError):
        wire.parse_plan(_plan(derivatives=[entry]), has_boxes=False)


def test_rendition_names_are_unique() -> None:
    with pytest.raises(wire.EnginePlanError) as err:
        wire.parse_plan(
            _plan(derivatives=[{'name': 'a', 'side': 64}, {'name': 'a', 'side': 128}]),
            has_boxes=False,
        )
    assert err.value.fault == 'bad_field'
