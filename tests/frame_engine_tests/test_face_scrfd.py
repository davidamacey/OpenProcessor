"""SCRFD post-processing of the ``faces`` branch (``frame_engine/faces/scrfd.py``) against the
reference's arithmetic (InsightFace ``v0.7`` ``model_zoo/scrfd.py``), written out here as the oracle."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from frame_engine.faces import scrfd as fs


def _ref_anchor_centers(rows: int, cols: int, stride: int) -> np.ndarray:
    """``SCRFD.forward``'s "solution-3" construction, verbatim."""
    ac = np.stack(list(np.mgrid[:rows, :cols][::-1]), axis=-1).astype(np.float32)
    ac = (ac * stride).reshape((-1, 2))
    return np.stack([ac] * 2, axis=1).reshape((-1, 2))


def _outputs(net: int, fires: dict[tuple[int, int], tuple[float, list[float], list[float]]]
             ) -> dict[str, np.ndarray]:  # fmt: skip
    """Zero outputs for a ``net`` input, with ``fires[(stride, anchor)] = (score, ltrb, kps10)``."""
    out = {}
    for s in fs.FPN_STRIDES:
        n = (net // s) ** 2 * fs.NUM_ANCHORS
        out[f'score_{s}'] = np.zeros((n, 1), np.float32)
        out[f'bbox_{s}'] = np.zeros((n, 4), np.float32)
        out[f'kps_{s}'] = np.zeros((n, 10), np.float32)
    for (s, a), (score, ltrb, kps) in fires.items():
        out[f'score_{s}'][a, 0] = score
        out[f'bbox_{s}'][a] = ltrb
        out[f'kps_{s}'][a] = kps
    return out


@pytest.mark.parametrize(('rows', 'cols', 'stride'), [(80, 80, 8), (40, 20, 16), (3, 7, 32)])
def test_anchor_centers_are_the_references(rows: int, cols: int, stride: int) -> None:
    """Catches a swapped x/y grid or anchors interleaved instead of repeated per centre."""
    np.testing.assert_array_equal(
        fs.anchor_centers(rows, cols, stride), _ref_anchor_centers(rows, cols, stride)
    )


def test_decode_box_and_landmarks_from_one_anchor() -> None:
    """Anchor 2 * (row * cols + col) + 1 at stride 16 is the centre (col * 16, row * 16); the
    distances and landmark offsets are in stride units. Catches a missing stride factor or a
    wrong anchor index."""
    row, col, cols = 5, 7, 640 // 16
    a = 2 * (row * cols + col) + 1
    kps = [0.5, -0.5, 1.0, 0.0, 0.0, 1.0, -1.0, 2.0, 2.0, 2.0]
    out = _outputs(640, {(16, a): (0.9, [1.0, 2.0, 3.0, 4.0], kps)})
    scores, boxes, lmk = fs.decode(out, 640, 640, 0.5)
    cx, cy = col * 16, row * 16
    np.testing.assert_allclose(scores, [0.9])
    np.testing.assert_allclose(boxes, [[cx - 16, cy - 32, cx + 48, cy + 64]])
    np.testing.assert_allclose(lmk[0], np.array(kps).reshape(5, 2) * 16 + [cx, cy])


def test_decode_threshold_is_inclusive_and_empty_is_typed() -> None:
    out = _outputs(
        640, {(8, 10): (0.5, [1, 1, 1, 1], [0] * 10), (32, 3): (0.4999, [1, 1, 1, 1], [0] * 10)}
    )
    scores, _, _ = fs.decode(out, 640, 640, 0.5)
    assert scores.tolist() == [0.5]
    s0, b0, k0 = fs.decode(_outputs(640, {}), 640, 640, 0.5)
    assert s0.shape == (0,)
    assert b0.shape == (0, 4)
    assert k0.shape == (0, 5, 2)


def test_nms_uses_the_references_plus_one_areas() -> None:
    """Boxes 0..9 and 2..11 (x), same y 0..9: IoU 64/116 = 0.55 with ``+1`` and 0.47 without;
    at threshold 0.5 the reference suppresses the second. Catches dropping the ``+1``."""
    boxes = np.array([[0, 0, 9, 9], [3, 0, 12, 9]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)
    # with +1: inter 7 * 10 = 70, union 100 + 100 - 70 = 130 -> 0.538; without: 6 * 9 = 54 / 108 = 0.5
    assert fs.nms(boxes, scores, 0.52).tolist() == [0]
    assert fs.nms(boxes, scores, 0.54).tolist() == [0, 1]


def test_nms_keeps_order_by_score_and_stable_ties() -> None:
    boxes = np.array([[0, 0, 10, 10], [100, 0, 110, 10], [200, 0, 210, 10]], np.float32)
    assert fs.nms(boxes, np.array([0.5, 0.7, 0.5], np.float32), 0.4).tolist() == [1, 0, 2]


def test_fit_is_the_references_letterbox_size() -> None:
    assert fs.fit(1000, 667, 640) == (640, 426)  # int(640 * 0.667)
    assert fs.fit(667, 1000, 640) == (426, 640)
    assert fs.fit(250, 250, 640) == (640, 640)


@pytest.mark.parametrize(
    ('w', 'h', 'scale', 'overlap'),
    [(5472, 3648, 0.35, 160), (1024, 683, 1.0, 160), (3000, 700, 0.5, 64)],
)
def test_tiles_cover_the_scaled_frame_with_the_overlap(
    w: int, h: int, scale: float, overlap: int
) -> None:
    """Every scaled pixel is in a window; neighbours share at least ``overlap``; ``inner`` marks
    exactly the window edges inside the frame. Catches a gap or a tile past the frame."""
    passes = fs.tile_passes(w, h, 640, scale, overlap)
    sw, sh = passes[0].scaled
    cover = np.zeros((sh, sw), np.int32)
    lefts = sorted({p.window[0] for p in passes})
    tops = sorted({p.window[1] for p in passes})
    for p in passes:
        left, top, ww, wh = p.window
        assert max(ww, wh) <= 640
        assert left + ww <= sw
        assert top + wh <= sh
        cover[top : top + wh, left : left + ww] += 1
        assert p.inner == (left > 0, top > 0, left + ww < sw, top + wh < sh)
    assert cover.min() >= 1
    for a, b in itertools.pairwise(lefts):
        assert a + 640 - b >= overlap
    for a, b in itertools.pairwise(tops):
        assert a + 640 - b >= overlap


def test_plan_passes_adds_tiles_only_when_finer_than_the_whole_pass() -> None:
    small = fs.plan_passes(250, 250, 640, coarse=256, tile_side=1920, overlap=160)
    assert [p.scaled for p in small] == [(640, 640), (256, 256)]
    big = fs.plan_passes(5472, 3648, 640, tile_side=1920, overlap=160)
    assert big[0].scaled == (640, 426)
    assert len(big) > 1
    assert all(p.scaled == (1920, 1280) for p in big[1:])


def test_a_photo_no_larger_than_the_tile_side_gets_no_tiles() -> None:
    """The rule that picks the scheme: tiles only when the long side exceeds ``tile_side``, so a
    2 MP photo is one detector call (whole + coarse) and a 20 MP photo is tiled at long side
    ``tile_side``. Catches tiles on every photo larger than the input."""
    two_mp = fs.plan_passes(1920, 1080, 640, coarse=256, tile_side=1920, overlap=160)
    assert [p.scaled for p in two_mp] == [(640, 360), (256, 144)]
    twenty = fs.plan_passes(5472, 3648, 640, coarse=256, tile_side=1920, overlap=160)
    assert all(p.scaled == (1920, 1280) for p in twenty[2:])
    assert len(twenty) == 2 + 12


def _seen_by(p: fs.DetPass, w: int, h: int, face: tuple[float, float, float, float]) -> tuple:
    """What an ideal detector answers for one pass: the face's part inside the window, in input
    pixels, score 0.9 (a face cut by the window is a cut box, as a real detector sees it)."""
    sx, sy = p.scaled[0] / w, p.scaled[1] / h
    left, top, ww, wh = p.window
    x1, y1, x2, y2 = (
        face[0] * sx - left,
        face[1] * sy - top,
        face[2] * sx - left,
        face[3] * sy - top,
    )
    cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, ww), min(y2, wh)
    if cx2 - cx1 < 4 or cy2 - cy1 < 4:
        return (
            np.zeros(0, np.float32),
            np.zeros((0, 4), np.float32),
            np.zeros((0, 5, 2), np.float32),
        )
    box = np.array([[cx1, cy1, cx2, cy2]], np.float32)
    kps = np.repeat(((box[:, :2] + box[:, 2:]) / 2)[:, None, :], 5, axis=1)
    return np.array([0.9], np.float32), box, kps


@pytest.mark.parametrize(
    ('face', 'why'),
    [
        ((1800.0, 1600.0, 1910.0, 1710.0), 'small face inside the overlap of two tiles'),
        ((1083.0, 900.0, 2565.0, 2382.0), 'face larger than the overlap, cut by every tile'),
        ((0.0, 1500.0, 110.0, 1610.0), 'small face on the frame edge'),
    ],
)
def test_a_face_across_a_tile_seam_is_found_once(face: tuple, why: str) -> None:
    """A 20 MP photo with tiles at 1920 and the whole pass: whatever the face's place, the merge
    returns it exactly once, never twice (two tiles) and never zero times (every tile cut it)."""
    w, h = 5472, 3648
    passes = fs.plan_passes(w, h, 640, tile_side=1920, overlap=160)
    k = 1920 / w
    edges = {p.window[0] for p in passes[1:]} | {p.window[0] + p.window[2] for p in passes[1:]}
    inner = [e for e in edges if 0 < e < 1920]
    assert any(face[0] * k < e < face[2] * k for e in inner) or face[0] == 0.0, why
    if 'every tile' in why:  # no tile holds it whole: only the whole pass can find it
        for p in passes[1:]:
            left, top, ww, wh = p.window
            inside = left <= face[0] * k and face[2] * k <= left + ww
            inside = inside and top <= face[1] * k and face[3] * k <= top + wh
            assert not inside
    parts = [fs.to_source(p, w, h, *_seen_by(p, w, h, face)) for p in passes]
    merged = fs.merge(parts, 0.4, 128)
    assert len(merged.scores) == 1, why
    np.testing.assert_allclose(merged.boxes[0], face, atol=4.0)


def test_to_source_maps_each_axis_by_its_own_scale_and_offset() -> None:
    """A tile at (left 300, top 100) of a frame scaled 0.5 x 0.5: input (10, 20) is source
    ((10 + 300) / 0.5, (20 + 100) / 0.5). The whole pass of a 1000 x 667 frame scales x by
    0.64 and y by 426 / 667. Catches the reference's single (height) scale for both axes."""
    p = fs.DetPass((500, 400), (300, 100, 200, 300), (True, True, False, False))
    _, b, k = fs.to_source(p, 1000, 800, np.array([0.9], np.float32),
                           np.array([[10, 20, 50, 60]], np.float32), np.full((1, 5, 2), 30, np.float32))  # fmt: skip
    np.testing.assert_allclose(b, [[620, 240, 700, 320]])
    np.testing.assert_allclose(k[0, 0], [660, 260])
    whole = fs.whole_pass(1000, 667, 640)
    _, b2, _ = fs.to_source(whole, 1000, 667, np.array([1.0], np.float32),
                            np.array([[640, 426, 640, 426]], np.float32), np.zeros((1, 5, 2), np.float32))  # fmt: skip
    np.testing.assert_allclose(b2[0], [1000, 667, 1000, 667], rtol=1e-6)


def test_to_source_drops_faces_cut_by_an_inner_tile_edge_only() -> None:
    p = fs.DetPass((2000, 640), (600, 0, 640, 640), (True, False, True, False))
    boxes = np.array(
        [[1, 50, 40, 90], [300, 0, 340, 30], [600, 50, 639, 90], [300, 600, 340, 640]], np.float32
    )
    s, _, _ = fs.to_source(
        p, 2000, 640, np.arange(4, dtype=np.float32), boxes, np.zeros((4, 5, 2), np.float32)
    )
    # left and right edges are inner: rows 0 and 2 go; top and bottom are the frame's: rows 1, 3 stay
    assert s.tolist() == [1.0, 3.0]


def test_merge_suppresses_across_passes_and_counts_the_overflow() -> None:
    a = (
        np.array([0.9, 0.6], np.float32),
        np.array([[0, 0, 10, 10], [100, 0, 110, 10]], np.float32),
        np.zeros((2, 5, 2), np.float32),
    )
    b = (np.array([0.8, 0.7, 0.5], np.float32),
         np.array([[0, 0, 10, 10], [200, 0, 210, 10], [300, 0, 310, 10]], np.float32), np.zeros((3, 5, 2), np.float32))  # fmt: skip
    faces = fs.merge([a, b], 0.4, 2)
    assert faces.scores.tolist() == pytest.approx([0.9, 0.7])
    assert faces.overflow == 2  # 0.6 and 0.5 survive NMS but are past the cap
    empty = fs.merge([], 0.4, 2)
    assert empty.boxes.shape == (0, 4)
    assert empty.overflow == 0
