"""``faces`` (round 1, settle) and ``face_embed`` (round 2): SCRFD on passes cut from the frame, then
ArcFace on five-point aligned faces sampled from the full-resolution frame.

``faces``: every pass of the photo (``frame_engine.faces.scrfd.plan_passes``: the whole frame in
the detector's input, the optional coarse square and tiles) is resampled from the resident frame
into its own zero-padded uint8 input, top-left as the reference letterbox; the passes go to the
detector in calls of :data:`PASSES_PER_CALL`, all submitted in round 1. ``settle`` collects them
and decodes on the host (anchor decode, NMS over every pass, the ``max_faces`` cap counted in
``face_overflow``) and leaves the faces on the photo. ``finish`` writes boxes and landmarks in
source pixels and the blur sums of each face box (``blur_sums``' arithmetic).

``face_embed``: each face's similarity to the ArcFace template, its crop warped by
``frame_engine.faces.kernels.WarpKernel`` from the full-resolution frame straight into its slot of
the staged recogniser batch (calls of the plan's ``batch`` faces, all submitted before any is
collected), the vectors L2-normalised as the face path's client does today. A face whose landmarks
give no transform is not embedded (``face_embedded``). A photo with no face makes no call.

Both models are the uint8-input builds (input ``images``, the normalisation in the graph): SCRFD
``(x - 127.5) / 128``, ArcFace ``(x - 127.5) / 127.5`` (InsightFace ``ArcFaceONNX`` for a graph
without its own normalisation, as ``w600k_r50`` is).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from frame_engine.branches.base import Branch, Call, DependencyFailedError, Photo
from frame_engine.faces import alignment, scrfd
from frame_engine.prep_geometry import blur_rect
from frame_engine.transport import ModelInput
from src.domain import frame_engine_wire as wire


if TYPE_CHECKING:
    from frame_engine.faces.kernels import WarpKernel
    from frame_engine.faces.scrfd import DetPass


FACE_INPUT = 'images'
FACE_OUTPUT = 'output'
#: Detector inputs per call: a 20 MP photo with tiles is a few tens of passes, never one call.
PASSES_PER_CALL = 8
#: The letterbox filter (plain bilinear as the reference's ``cv2.resize``, or antialiased); chosen
#: on WIDER FACE (``scripts/research/face_engine``).
ANTIALIAS = False


class Faces(Branch):
    status = 'faces'

    def __init__(self, spec: wire.FaceSpec) -> None:
        self.spec = spec
        self.passes: list[DetPass] = []
        self.calls: list[Call] = []
        self.faces: scrfd.Faces | None = None

    def round1(self, photo: Photo) -> None:
        w, h = photo.size
        s = self.spec
        net = s.input_size
        self.passes = scrfd.plan_passes(
            w, h, net, coarse=s.coarse, tile_side=s.tile_side, overlap=s.overlap
        )
        for k in range(0, len(self.passes), PASSES_PER_CALL):
            chunk = self.passes[k : k + PASSES_PER_CALL]
            tensor = ModelInput(FACE_INPUT, 'UINT8', (len(chunk), 3, net, net))
            staged = photo.calls.stage(tensor.nbytes, 'face_letterbox')
            with photo.clock.stage('face_letterbox'):
                canvas = staged.device.reshape(tensor.shape)
                canvas.fill(0)
                for i, p in enumerate(chunk):
                    _, _, ww, wh = p.window
                    photo.kernels.resample_into(
                        photo.frame, (0, 0, w, h), p.scaled, canvas[i, :, :wh, :ww], p.window,
                        antialias=ANTIALIAS,
                    )  # fmt: skip
            self.calls.append(
                photo.calls.submit(
                    self, 'face_detect', staged, s.detector, tensor, scrfd.OUTPUT_NAMES
                )
            )

    def settle(self, photo: Photo) -> None:
        answers = [photo.calls.result(c) for c in self.calls]
        w, h = photo.size
        net = self.spec.input_size
        with photo.clock.stage('face_detect'):
            parts = []
            for k, ans in enumerate(answers):
                for i, p in enumerate(self.passes[k * PASSES_PER_CALL : (k + 1) * PASSES_PER_CALL]):
                    one = {name: ans[name][i] for name in scrfd.OUTPUT_NAMES}
                    parts.append(
                        scrfd.to_source(p, w, h, *scrfd.decode(one, net, net, self.spec.score_min))
                    )
            self.faces = photo.faces = scrfd.merge(parts, self.spec.nms, self.spec.max_faces)

    def finish(self, photo: Photo) -> None:
        if self.faces is None:
            raise RuntimeError('faces finished before settle')
        f = self.faces
        photo.out[wire.FACE_BOXES] = f.boxes
        photo.out[wire.FACE_SCORES] = f.scores
        photo.out[wire.FACE_LANDMARKS] = f.landmarks.reshape(-1, 10)
        photo.out[wire.FACE_OVERFLOW] = np.array([f.overflow], np.int32)
        with photo.clock.stage('face_blur'):
            photo.out[wire.FACE_BLUR_SUMS] = _blur_sums(photo, f.boxes)


def _blur_sums(photo: Photo, boxes: np.ndarray) -> np.ndarray:
    """``[faces, 3]`` (pixels, sum, sum of squares) of each face box clamped to the frame; a box
    with nothing inside is ``(0, 0, 0)``."""
    w, h = photo.size
    clamped = [blur_rect((float(b[0]), float(b[1]), float(b[2]), float(b[3])), w, h) for b in boxes]
    rects = [r for r in clamped if r is not None]
    sums = iter(cp.asnumpy(photo.kernels.lap_sums(photo.frame, rects)))
    rows = [
        (0, 0, 0) if r is None else ((r[2] - r[0]) * (r[3] - r[1]), *next(sums)) for r in clamped
    ]
    return np.array(rows, np.int64).reshape(-1, 3)


class FaceEmbed(Branch):
    status = 'face_embed'
    needs = ('faces',)

    def __init__(self, spec: wire.FaceSpec, warp: WarpKernel) -> None:
        self.spec = spec
        self.warp = warp
        self.embedded: np.ndarray | None = None
        self.calls: list[Call] = []

    def round2(self, photo: Photo) -> None:
        if photo.faces is None:
            raise DependencyFailedError('no faces')
        inv, taps, ok = alignment.warp_params(photo.faces.landmarks)
        self.embedded = ok
        rows = np.flatnonzero(ok)
        for k in range(0, len(rows), self.spec.batch):
            chunk = rows[k : k + self.spec.batch]
            size = alignment.ARCFACE_SIZE
            tensor = ModelInput(FACE_INPUT, 'UINT8', (len(chunk), 3, size, size))
            staged = photo.calls.stage(tensor.nbytes, 'face_align')
            with photo.clock.stage('face_align'):
                self.warp.warp_into(
                    photo.frame, inv[chunk], taps[chunk], staged.device.reshape(tensor.shape)
                )
            self.calls.append(
                photo.calls.submit(
                    self, 'face_embed', staged, self.spec.recogniser, tensor, (FACE_OUTPUT,)
                )
            )

    def finish(self, photo: Photo) -> None:
        if self.embedded is None:
            raise RuntimeError('face_embed finished before round 2')
        rows = [photo.calls.result(c)[FACE_OUTPUT].astype(np.float32) for c in self.calls]
        vectors = np.concatenate(rows, axis=0) if rows else np.zeros((0, 0), np.float32)
        norms = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-10)
        photo.out[wire.FACE_EMBEDDINGS] = vectors / norms
        photo.out[wire.FACE_EMBEDDED] = self.embedded
