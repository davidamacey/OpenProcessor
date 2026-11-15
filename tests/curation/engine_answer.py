"""What a frame engine answers for one photo: the ONE builder of a fake engine's answer.

Two fakes use it: the in-process pool of the ingest unit tests (``tests/curation/engine_fakes.py``,
fixed numbers) and the live harness's engine service (``docker/test/fake_frame_engine.py``, which
passes the detections, vectors and crops it computed from the photo). The answer's structure is
here once: the branch states, the letterbox, one blur row per detection, only the rows the plan
embeds, only the outputs the plan asks for (:func:`~src.domain.frame_engine_wire.expected_outputs`).

It imports only numpy, the wire module and the engine's own result type, so the harness container
(the API image, with ``frame_engine/`` mounted) runs it too.
"""

from __future__ import annotations

import numpy as np

from frame_engine.messages import EngineResult
from src.domain import frame_engine_wire as wire


#: The detector canvas the fixed numbers of the unit-test fake are written for.
DEFAULT_CANVAS = 320
#: One detection of the unit-test fake, in canvas pixels: ``x1, y1, x2, y2, score, class``.
DEFAULT_BOX = (16.0, 16.0, 192.0, 192.0, 0.9)
#: Blur sums ``[n, sum, squares]`` the fakes report: the frame's variance is 5.0, a box's 2.0.
FRAME_BLUR = (100, 0, 500)
BOX_BLUR = (10, 10, 30)


def _axis(dim: int, index: int, value: float) -> np.ndarray:
    out = np.zeros(dim, np.float32)
    out[index] = value
    return out


def letterbox_of(width: int, height: int, canvas: int) -> tuple[float, float, float]:
    """``(scale, pad_x, pad_y)`` of a ``width`` x ``height`` photo centred on a ``canvas`` square."""
    scale = min(canvas / width, canvas / height)
    return scale, (canvas - round(width * scale)) // 2, (canvas - round(height * scale)) // 2


def failed_answer(plan: wire.EnginePlan, status: str) -> EngineResult:
    """The answer for a photo the engine could not read: ``status`` is one of
    :data:`src.domain.decode_route.PREP_STATUSES`, no branch ran."""
    branches: dict[str, tuple[wire.BranchState, str]] = dict.fromkeys(
        plan.branches, ('error', wire.IMAGE_FAILED)
    )
    return EngineResult(status, wire.DECODED_NONE, branches)


def engine_answer(
    plan: wire.EnginePlan,
    width: int,
    height: int,
    *,
    route: str = 'gpu',
    failed_branch: str | None = None,
    canvas: int = DEFAULT_CANVAS,
    det_class: int = 1,
    embed_dim: int = 3,
    detections: np.ndarray | None = None,
    whole: np.ndarray | None = None,
    crop_vectors: np.ndarray | None = None,
    crop_jpegs: list[bytes] | None = None,
    jpeg: bytes = b'\xff\xd8crop',
    box_count: int | None = None,
) -> EngineResult:
    """The answer for a decoded ``width`` x ``height`` photo under ``plan``.

    ``detections`` (rows ``x1, y1, x2, y2, score, class`` in canvas pixels), ``whole`` (the frame
    vector), ``crop_vectors`` (one row per detection or box, before the plan picks rows) and
    ``crop_jpegs`` (one per detection or box) replace the fixed numbers. ``box_count`` is the
    number of stored boxes the caller sent: the plan has no ``detect`` then, so no detection
    tensor is answered, and the rows are the boxes'."""
    branches: dict[str, tuple[wire.BranchState, str]] = dict.fromkeys(plan.branches, ('ok', ''))
    if failed_branch is not None:
        branches[failed_branch] = ('error', wire.MODEL_ERROR)
    if detections is None:
        detections = np.array([[*DEFAULT_BOX, det_class]], np.float32)
    detections = np.asarray(detections, np.float32).reshape(-1, 6)
    n = detections.shape[0] if box_count is None else box_count
    classes = detections[:, 5].astype(np.int32) if box_count is None else None
    out: dict[str, np.ndarray | list[bytes]] = {
        wire.SIZE: np.array([width, height], np.int32),
        wire.LETTERBOX: np.array(letterbox_of(width, height, canvas), np.float64),
        wire.DET_BOXES: detections[:, :4],
        wire.DET_SCORES: detections[:, 4],
        wire.DET_CLASSES: detections[:, 5].astype(np.int32),
        wire.WHOLE_EMBEDDING: _axis(embed_dim, embed_dim - 1, 2.0) if whole is None else whole,
        wire.BLUR_SUMS: np.array([FRAME_BLUR, *([BOX_BLUR] * n)], np.int64),
        wire.CROP_JPEGS: crop_jpegs if crop_jpegs is not None else [jpeg] * n,
    }
    if plan.runs('embed_crops'):
        # as the engine does: only the picked rows are embedded, their rows come first
        picked = np.ones(n, np.bool_)
        if plan.embed_classes is not None and classes is not None:
            picked = np.isin(classes, sorted(plan.embed_classes))
        dim = len(out[wire.WHOLE_EMBEDDING])
        rows = (
            np.tile(_axis(dim, min(1, dim - 1), 3.0), (n, 1))
            if crop_vectors is None
            else np.asarray(crop_vectors, np.float32).reshape(n, dim)
        )
        out[wire.CROP_EMBEDDINGS] = rows[picked].astype(np.float32)
        out[wire.CROP_EMBEDDED] = picked
    wanted = {wire.SIZE, *wire.expected_outputs(plan)}
    return EngineResult(
        wire.STATUS_OK, route, branches, {k: v for k, v in out.items() if k in wanted}
    )
