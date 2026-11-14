"""SCRFD post-processing on the host (numpy): passes, anchor decode, NMS across passes.

A photo is seen by the detector through one or more *passes*. A pass is a window of the frame
resampled by a scale, placed at the top-left of the detector's square input with zero padding
(InsightFace ``SCRFD.detect``). Three kinds:

* the whole frame fitted into the input (the reference's letterbox);
* tiles: the frame at a larger scale, cut into overlapping windows of the input size, so a small
  face in a large photo reaches the detector at a size it finds;
* a coarse pass: the whole frame fitted into a smaller square inside the zero-padded input, for
  faces too large for the anchors at the input size (a portrait that fills the photo).

The decode is the reference's (``model_zoo/scrfd.py`` at InsightFace ``v0.7``): per stride,
anchor centres on the stride grid, two anchors per centre, distances times the stride, score
``>=`` threshold. Two departures, both measured on WIDER FACE (``scripts/research/face_engine``):

* each axis is mapped back by its own scale; the reference divides both by the height's scale, so
  a landscape photo's boxes drift by up to a few pixels toward its right edge;
* NMS runs once over the detections of every pass, in source pixels, with the reference's
  ``+1`` area convention.

A tile drops the detections that touch one of its edges inside the frame: the overlap is chosen
so such a face is whole in a neighbouring tile or in the whole-frame pass.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

FPN_STRIDES = (8, 16, 32)
NUM_ANCHORS = 2
#: Output tensor names of the batch-patched SCRFD (``export/scrfd_onnx.py``), per stride.
SCORE_NAMES = tuple(f'score_{s}' for s in FPN_STRIDES)
BBOX_NAMES = tuple(f'bbox_{s}' for s in FPN_STRIDES)
KPS_NAMES = tuple(f'kps_{s}' for s in FPN_STRIDES)
OUTPUT_NAMES = SCORE_NAMES + BBOX_NAMES + KPS_NAMES
#: A detection this close to an inner tile edge (input pixels) is left to the neighbour.
EDGE_MARGIN = 2.0


class DetPass(NamedTuple):
    """The frame resampled to ``scaled`` (w, h) and the ``window`` (left, top, w, h) of it that
    sits at the input's top-left. ``inner`` marks the window edges (left, top, right, bottom)
    that lie inside the scaled frame, where a tile cuts faces."""

    scaled: tuple[int, int]
    window: tuple[int, int, int, int]
    inner: tuple[bool, bool, bool, bool]


class Faces(NamedTuple):
    """Detections in source pixels, best score first: boxes ``[n, 4]``, scores ``[n]``,
    landmarks ``[n, 5, 2]``; ``overflow`` counts the ones past the cap (dropped)."""

    boxes: np.ndarray
    scores: np.ndarray
    landmarks: np.ndarray
    overflow: int


def fit(w: int, h: int, side: int) -> tuple[int, int]:
    """The reference's letterbox size: the longer side to ``side``, the other truncated."""
    if w <= 0 or h <= 0:
        raise ValueError(f'degenerate image size: ({w}, {h})')
    ratio = h / w
    if ratio > 1.0:
        return max(1, int(side / ratio)), side
    return side, max(1, int(side * ratio))


def whole_pass(w: int, h: int, side: int) -> DetPass:
    sw, sh = fit(w, h, side)
    return DetPass((sw, sh), (0, 0, sw, sh), (False, False, False, False))


def _starts(total: int, net: int, overlap: int) -> list[int]:
    if total <= net:
        return [0]
    n = -(-(total - overlap) // (net - overlap))
    return [round(k * (total - net) / (n - 1)) for k in range(n)]


def tile_passes(w: int, h: int, net: int, scale: float, overlap: int) -> list[DetPass]:
    """Windows of ``net`` covering the frame at ``scale``, neighbours sharing at least
    ``overlap`` input pixels; a frame that fits at that scale is one window."""
    if not 0 <= overlap < net:
        raise ValueError(f'overlap {overlap} outside [0, {net})')
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    out = []
    for top in _starts(sh, net, overlap):
        for left in _starts(sw, net, overlap):
            ww, wh = min(net, sw - left), min(net, sh - top)
            inner = (left > 0, top > 0, left + ww < sw, top + wh < sh)
            out.append(DetPass((sw, sh), (left, top, ww, wh), inner))
    return out


def plan_passes(
    w: int, h: int, net: int, *, coarse: int = 0, tile_side: int = 0, overlap: int = 0
) -> list[DetPass]:
    """The passes of one photo: the whole frame in ``net``; with ``coarse``, the whole frame in a
    ``coarse`` square as well; with ``tile_side``, tiles of the frame scaled so its longer side is
    ``tile_side`` (never above the source size), when that is finer than the whole pass."""
    passes = [whole_pass(w, h, net)]
    if coarse:
        passes.append(whole_pass(w, h, min(coarse, net)))
    if tile_side:
        scale = min(1.0, tile_side / max(w, h))
        if round(max(w, h) * scale) > max(passes[0].scaled):
            passes.extend(tile_passes(w, h, net, scale, overlap))
    return passes


@lru_cache(maxsize=32)
def anchor_centers(rows: int, cols: int, stride: int) -> np.ndarray:
    """``[rows * cols * NUM_ANCHORS, 2]`` (x, y) centres in input pixels, the reference's order."""
    gy, gx = np.mgrid[:rows, :cols]
    grid = np.stack([gx, gy], axis=-1).astype(np.float32)
    centers = (grid * stride).reshape(-1, 2)
    centers = np.repeat(centers, NUM_ANCHORS, axis=0)
    centers.setflags(write=False)
    return centers


def decode(
    outputs: Mapping[str, np.ndarray], net_w: int, net_h: int, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One input's detections at or above ``threshold``, unsorted, in input pixels:
    scores ``[n]``, boxes ``[n, 4]``, landmarks ``[n, 5, 2]``. ``outputs`` maps each name of
    :data:`OUTPUT_NAMES` to that input's ``[anchors, d]`` rows."""
    scores, boxes, kpss = [], [], []
    for stride, sn, bn, kn in zip(FPN_STRIDES, SCORE_NAMES, BBOX_NAMES, KPS_NAMES, strict=True):
        s = np.asarray(outputs[sn]).reshape(-1)
        pos = np.flatnonzero(s >= threshold)
        if pos.size == 0:
            continue
        c = anchor_centers(net_h // stride, net_w // stride, stride)[pos]
        d = np.asarray(outputs[bn]).reshape(-1, 4)[pos] * stride
        k = np.asarray(outputs[kn]).reshape(-1, 10)[pos].reshape(-1, 5, 2) * stride
        scores.append(s[pos])
        boxes.append(np.concatenate([c - d[:, :2], c + d[:, 2:]], axis=1))
        kpss.append(k + c[:, None, :])
    if not scores:
        return (
            np.zeros(0, np.float32),
            np.zeros((0, 4), np.float32),
            np.zeros((0, 5, 2), np.float32),
        )
    return np.concatenate(scores), np.concatenate(boxes), np.concatenate(kpss)


def to_source(
    p: DetPass, w: int, h: int, scores: np.ndarray, boxes: np.ndarray, kps: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One pass's detections mapped to source pixels, without those cut by an inner tile edge."""
    left, top, ww, wh = p.window
    keep = np.ones(len(scores), np.bool_)
    il, it, ir, ib = p.inner
    if il:
        keep &= boxes[:, 0] > EDGE_MARGIN
    if it:
        keep &= boxes[:, 1] > EDGE_MARGIN
    if ir:
        keep &= boxes[:, 2] < ww - EDGE_MARGIN
    if ib:
        keep &= boxes[:, 3] < wh - EDGE_MARGIN
    sx, sy = p.scaled[0] / w, p.scaled[1] / h
    off = np.array([left, top], np.float32)
    scale = np.array([sx, sy], np.float32)
    b = (boxes[keep].reshape(-1, 2, 2) + off) / scale
    k = (kps[keep] + off) / scale
    return scores[keep], b.reshape(-1, 4), k


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy NMS, the reference's ``SCRFD.nms`` (``+1`` areas, stable descending order)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(-scores, kind='stable')
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]) + 1)
        inter = w * h
        order = rest[inter / (areas[i] + areas[rest] - inter) <= threshold]
    return np.asarray(keep, np.int64)


def merge(
    parts: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]], nms_threshold: float, cap: int
) -> Faces:
    """NMS over every pass's source-pixel detections; the best ``cap`` are kept, the rest counted."""
    parts = [p for p in parts if len(p[0])]
    if not parts:
        return Faces(np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                     np.zeros((0, 5, 2), np.float32), 0)  # fmt: skip
    scores = np.concatenate([p[0] for p in parts]).astype(np.float32)
    boxes = np.concatenate([p[1] for p in parts]).astype(np.float32)
    kps = np.concatenate([p[2] for p in parts]).astype(np.float32)
    keep = nms(boxes, scores, nms_threshold)
    return Faces(boxes[keep[:cap]], scores[keep[:cap]], kps[keep[:cap]], max(0, len(keep) - cap))
