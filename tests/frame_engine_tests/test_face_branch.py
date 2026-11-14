"""The ``faces`` and ``face_embed`` branches in the engine (``frame_engine/branches/faces.py``), driven
by their plan against the engine tests' stand-ins: the fake model server answers ``scrfd`` with two
faces (centres 160, 160 and 384, 320 of the 640 input; scores 0.9, 0.8) and the recogniser with
8-wide vectors read from the staged crops."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.frame_engine_tests import test_engine as te


eng = te.eng  # the engine tests' fixture: the real engine against their stand-ins
FACES, H, W = te.FACES, te.H, te.W


FACE_STAGES = {'face_letterbox', 'face_detect', 'face_align', 'face_embed', 'face_blur'}


def _faces_plan(**faces: Any) -> Any:
    return te._plan(['faces', 'face_embed'], detector=None, embed=None, crops='none',
                 faces={**FACES, **faces})  # fmt: skip


def test_faces_and_vectors_come_from_one_detector_call_and_one_recogniser_call(eng: Any) -> None:
    """The whole pass is the reference letterbox (top-left, 96 x 64 to 640 x 426); boxes come back
    in source pixels by each axis' own scale. Catches a second detector call for a plan without
    tiles, an unscaled box, unnormalised vectors and a missing output."""
    engine = eng()
    r = te._run(engine, _faces_plan())
    assert engine.transport.calls == [('scrfd', [1, 3, 640, 640]), ('arc', [2, 3, 112, 112])]
    assert r.branches == {'faces': ('ok', ''), 'face_embed': ('ok', '')}
    sx, sy = 640 / W, 426 / H
    np.testing.assert_allclose(
        r.outputs['face_boxes'][0], [96 / sx, 96 / sy, 224 / sx, 224 / sy], rtol=1e-5
    )
    assert r.outputs['face_scores'].tolist() == pytest.approx([0.9, 0.8])
    assert r.outputs['face_landmarks'].shape == (2, 10)
    assert r.outputs['face_blur_sums'].shape == (2, 3)
    assert r.outputs['face_overflow'].tolist() == [0]
    vec = r.outputs['face_embeddings']
    assert vec.shape == (2, 8)
    np.testing.assert_allclose(np.linalg.norm(vec, axis=1), 1.0, rtol=1e-6)
    assert r.outputs['face_embedded'].tolist() == [True, True]
    assert set(te._seen(engine)) >= FACE_STAGES
    assert engine.transport.staged == 0


def test_a_photo_without_a_face_costs_one_detector_call(eng: Any) -> None:
    engine = eng()
    real = engine.transport.collect

    def nobody(p: Any) -> Any:
        out = real(p)
        return {k: np.zeros_like(v) for k, v in out.items()} if p.model == 'scrfd' else out

    engine.transport.collect = nobody
    r = te._run(engine, _faces_plan())
    assert engine.transport.models() == ['scrfd']
    assert r.branches == {'faces': ('ok', ''), 'face_embed': ('ok', '')}
    assert r.outputs['face_boxes'].shape == (0, 4)
    assert r.outputs['face_embeddings'].shape[0] == 0
    assert r.outputs['face_embedded'].tolist() == []


def test_recogniser_calls_are_bounded_by_the_plan_batch(eng: Any) -> None:
    engine = eng()
    te._run(engine, _faces_plan(batch=1))
    assert engine.transport.calls[1:] == [('arc', [1, 3, 112, 112])] * 2


def test_faces_past_the_cap_are_counted_not_embedded(eng: Any) -> None:
    engine = eng()
    r = te._run(engine, _faces_plan(max_faces=1))
    assert r.outputs['face_scores'].tolist() == pytest.approx([0.9])
    assert r.outputs['face_overflow'].tolist() == [1]
    assert engine.transport.calls[1:] == [('arc', [1, 3, 112, 112])]


def test_passes_go_in_bounded_calls(eng: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the coarse pass a photo has two detector inputs: one call of two, or two calls of one
    when a call holds one pass. Catches a per-photo pass count reaching one call."""
    engine = eng()
    te._run(engine, _faces_plan(coarse=256))
    assert engine.transport.calls[0] == ('scrfd', [2, 3, 640, 640])
    import frame_engine.branches.faces as fb

    monkeypatch.setattr(fb, 'PASSES_PER_CALL', 1)
    engine.transport.calls.clear()
    te._run(engine, _faces_plan(coarse=256))
    assert engine.transport.models()[:2] == ['scrfd', 'scrfd']


def test_a_failed_detector_fails_faces_and_what_needs_it_only(eng: Any) -> None:
    from frame_engine.transport import ModelCallError

    engine = eng()
    engine.transport.fail['scrfd'] = ModelCallError('scrfd', 'StatusCode.INTERNAL', 'boom')
    plan = te._plan(['detect', 'faces', 'face_embed'], embed=None, crops='none', faces=FACES)
    r = te._run(engine, plan)
    assert r.status == 'ok'
    assert r.branches == {
        'detect': ('ok', ''),
        'faces': ('error', 'model_error'),
        'face_embed': ('error', 'dependency_failed'),
    }
    assert 'arc' not in engine.transport.models()
    assert 'det_boxes' in r.outputs
    assert not {'face_boxes', 'face_embeddings'} & set(r.outputs)
    assert engine.transport.staged == 0


def test_a_failed_recogniser_fails_face_embed_only(eng: Any) -> None:
    from frame_engine.transport import ModelCallError

    engine = eng()
    engine.transport.fail['arc'] = ModelCallError('arc', 'StatusCode.INTERNAL', 'boom')
    r = te._run(engine, _faces_plan())
    assert r.branches == {'faces': ('ok', ''), 'face_embed': ('error', 'model_error')}
    assert r.outputs['face_boxes'].shape == (2, 4)
    assert 'face_embeddings' not in r.outputs
