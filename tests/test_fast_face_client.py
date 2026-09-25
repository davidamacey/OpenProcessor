"""
Unit test for DF3: /faces/* `orig_shape` must report the TRUE original image
dimensions, not the 1024-capped working size, while box/landmark
normalization must stay correct against the original.

No Triton/GPU dependency: `_call_scrfd` and `_call_arcface` are monkeypatched
to synthetic outputs so the test runs on the CPU decode/normalize path only.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.clients.fast_face_client import FastFaceClient


@pytest.fixture
def face_client(monkeypatch: pytest.MonkeyPatch) -> FastFaceClient:
    # Bypass TritonClientManager entirely; recognize() only needs
    # _call_scrfd/_call_arcface, which we stub out below.
    client = FastFaceClient.__new__(FastFaceClient)
    client.client = None
    client.scrfd_model = 'scrfd_10g_bnkps'
    client.arcface_model = 'arcface_w600k_r50'
    return client


def _encode_jpeg(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.jpg', img_bgr)
    assert ok
    return buf.tobytes()


def test_orig_shape_reports_true_original_dims(
    face_client: FastFaceClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 2000x1500 input must report orig_shape (1500, 2000), the TRUE size,
    not the 1024-capped working size, with boxes still normalized correctly.
    """
    true_h, true_w = 1500, 2000
    img = np.full((true_h, true_w, 3), 200, dtype=np.uint8)
    # A synthetic face-like box centered at pixel (1000, 750) in the ORIGINAL
    # image; roughly a 300x300 px box.
    orig_box_px = np.array([[850.0, 600.0, 1150.0, 900.0]], dtype=np.float32)
    orig_landmarks_px = np.array(
        [
            [
                [950.0, 700.0],
                [1050.0, 700.0],
                [1000.0, 750.0],
                [960.0, 820.0],
                [1040.0, 820.0],
            ]
        ],
        dtype=np.float32,
    )

    # Determine the cap scale the same way recognize() will (1024 / max dim).
    cap_scale = 1024 / max(true_h, true_w)

    def fake_decode(raw_outputs, det_scale, det_thresh, nms_thresh, max_faces):
        # decode_scrfd_outputs is called on the (possibly capped) working
        # image; det_scale tells us how the caller scaled pixel coords back
        # up. Since the working image is a uniform downscale of the true
        # original (aspect ratio preserved), applying the SAME cap scale to
        # our "true" boxes gives coordinates in the capped working space.
        boxes = orig_box_px * cap_scale
        landmarks = orig_landmarks_px * cap_scale
        scores = np.array([0.99], dtype=np.float32)
        return boxes, scores, landmarks

    def fake_call_scrfd(blob):
        return {}

    def fake_call_arcface(faces):
        return np.ones((len(faces), 512), dtype=np.float32) / np.sqrt(512)

    monkeypatch.setattr('src.clients.fast_face_client.decode_scrfd_outputs', fake_decode)
    monkeypatch.setattr(face_client, '_call_scrfd', fake_call_scrfd)
    monkeypatch.setattr(face_client, '_call_arcface', fake_call_arcface)

    image_bytes = _encode_jpeg(img)
    result = face_client.recognize(image_bytes, confidence=0.5)

    assert result['status'] == 'success'
    assert result['num_faces'] == 1

    # DF3: orig_shape must be the TRUE original size, not the capped size.
    reported_h, reported_w = result['orig_shape']
    assert (reported_h, reported_w) == (true_h, true_w)
    assert max(reported_h, reported_w) > 1024  # sanity: really is uncapped

    # Boxes must still be normalized correctly against the TRUE original,
    # i.e. dividing the true pixel box by the true dimensions.
    expected_box_norm = orig_box_px[0] / np.array([true_w, true_h, true_w, true_h])
    got_box_norm = np.array(result['face_boxes'][0])
    assert np.allclose(got_box_norm, expected_box_norm, atol=1e-3)

    # Landmarks likewise.
    expected_lmk_norm = orig_landmarks_px[0] / np.array([true_w, true_h])
    got_lmk_flat = np.array(result['face_landmarks'][0]).reshape(5, 2)
    assert np.allclose(got_lmk_flat, expected_lmk_norm, atol=1e-3)


def test_orig_shape_uncapped_image_reports_actual_size(
    face_client: FastFaceClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An image already under the 1024 cap should report its own true size
    (regression guard: this already worked before the fix)."""
    true_h, true_w = 480, 640
    img = np.full((true_h, true_w, 3), 128, dtype=np.uint8)

    def fake_decode(*_args, **_kwargs):
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 5, 2), dtype=np.float32),
        )

    monkeypatch.setattr('src.clients.fast_face_client.decode_scrfd_outputs', fake_decode)
    monkeypatch.setattr(face_client, '_call_scrfd', lambda blob: {})  # noqa: ARG005

    result = face_client.recognize(_encode_jpeg(img), confidence=0.5)
    assert result['num_faces'] == 0
    assert result['orig_shape'] == (true_h, true_w)
