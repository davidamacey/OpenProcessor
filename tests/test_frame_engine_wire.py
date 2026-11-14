"""Frame engine protocol version 1: the plan is taken exactly as written or refused."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.domain import frame_engine_wire as wire
from src.domain.processing_plan import BRANCHES


DETECTOR = {'model': 'yolov11_small_trt_end2end', 'input_size': 640, 'letterbox_fill': 114}
EMBED = {'model': 'pe_image_encoder', 'size': 336, 'batch': 32}
FACES = {'detector': 'scrfd_u8', 'input_size': 640, 'recogniser': 'arcface_u8', 'batch': 32,
         'score_min': 0.5, 'nms': 0.4, 'max_faces': 128, 'coarse': 256, 'tile_side': 1920,
         'overlap': 160}  # fmt: skip


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
    assert set(wire.ENGINE_BRANCHES) >= wire.BUILT_BRANCHES


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
        (_plan(branches=['ocr'], crops='none', if_detected={'ocr': [2]}), 'missing_dependency'),
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


def test_unbuilt_branches_and_derivatives_are_reported_unsupported() -> None:
    plan = wire.parse_plan(
        _plan(
            branches=['detect', 'faces', 'face_embed', 'ocr'],
            embed=None,
            faces=FACES,
            crops='none',
            if_detected={'ocr': [0]},
            derivatives=[{'name': 'thumb', 'side': 256}],
        ),
        has_boxes=False,
    )
    assert plan.unsupported() == ('ocr', 'derivatives')
    assert plan.reported() == ('detect', 'faces', 'face_embed', 'ocr', 'derivatives')
    assert plan.if_detected == {'ocr': (0,)}


def test_the_faces_plan_parses_and_round_trips() -> None:
    plan = wire.parse_plan(_plan(branches=['faces', 'face_embed'], detector=None, embed=None,
                                 crops='none', faces=FACES), has_boxes=False)  # fmt: skip
    assert plan.faces == wire.FaceSpec(
        'scrfd_u8', 640, 'arcface_u8', 32, 0.5, 0.4, 128, 256, 1920, 160
    )
    assert wire.parse_plan(plan.to_json(), has_boxes=False) == plan
    lean = {k: v for k, v in FACES.items() if k not in ('coarse', 'tile_side', 'overlap')}
    plan = wire.parse_plan(_plan(branches=['faces'], detector=None, embed=None, crops='none',
                                 faces=lean), has_boxes=False)  # fmt: skip
    assert plan.faces is not None
    assert (plan.faces.coarse, plan.faces.tile_side, plan.faces.overlap) == (0, 0, 0)
    assert wire.expected_outputs(plan) == (
        'face_boxes',
        'face_scores',
        'face_landmarks',
        'face_blur_sums',
        'face_overflow',
    )


@pytest.mark.parametrize(
    ('faces', 'branches', 'fault'),
    [
        (None, ['faces'], 'missing_field'),
        (FACES, ['detect'], 'bad_field'),
        ({**FACES, 'score_min': 1.5}, ['faces'], 'bad_field'),
        ({**FACES, 'nms': '0.4'}, ['faces'], 'bad_field'),
        ({**FACES, 'input_size': 600}, ['faces'], 'bad_field'),
        ({**FACES, 'coarse': 16}, ['faces'], 'bad_field'),
        ({**FACES, 'overlap': 640}, ['faces'], 'bad_field'),
        ({**FACES, 'max_faces': 0}, ['faces'], 'bad_field'),
        ({**FACES, 'tiles': 4}, ['faces'], 'unknown_field'),
        ({k: v for k, v in FACES.items() if k != 'recogniser'}, ['faces'], 'missing_field'),
    ],
)
def test_a_bad_faces_object_is_refused(faces: Any, branches: list[str], fault: str) -> None:
    """Catches a faces branch without its settings, settings for a branch not asked for, and a
    threshold, size or overlap the branch cannot use."""
    with pytest.raises(wire.EnginePlanError) as err:
        wire.parse_plan(
            _plan(branches=branches, embed=None, crops='none', faces=faces), has_boxes=False
        )
    assert err.value.fault == fault


def test_a_condition_on_a_built_branch_is_refused_until_the_engine_runs_conditions() -> None:
    """The orchestrator has no conditional path yet: a plan asking for faces only when a class
    is detected would get faces on every photo. Refused (fail closed), never silently run."""
    with pytest.raises(wire.EnginePlanError) as err:
        wire.parse_plan(_plan(branches=['detect', 'faces'], embed=None, crops='none', faces=FACES,
                              if_detected={'faces': [0]}), has_boxes=False)  # fmt: skip
    assert (err.value.fault, err.value.detail) == ('bad_field', 'if_detected.faces')


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
