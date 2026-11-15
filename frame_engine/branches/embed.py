"""``embed_frame`` (round 1) and ``embed_crops`` (round 2): encoder squares cut from the frame.

The two encoders are models of their own: ``embed_frame`` runs ``plan.embed``, ``embed_crops``
runs ``plan.crop_embed``, each with its own size and recipe.

The whole-frame square is Pillow ``BILINEAR`` (antialiased), shorter edge to the encoder size,
centre crop (``src.utils.pe_preprocess.whole_frame_rgb_from_image``). A crop square is the whole
rectangle (``crop_pil``'s) scaled to fit the square with its aspect kept, on a square of the
plan's ``fill`` (the letterbox geometry of :func:`~src.domain.image_geometry.letterbox_geometry`
and the resample rule beside it, antialiased when the box is larger than the square and plain
bilinear when it is smaller, the ones the API's host copy uses). The crop squares go through the
encoder in calls of the plan's batch, each call its own slice of the region; every call is
submitted before any is collected, within the photo's window
(:class:`~frame_engine.branches.base.PhotoCalls`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from frame_engine.branches.base import Branch, Call, DependencyFailedError, Photo
from frame_engine.transport import ModelInput
from src.domain import frame_engine_wire as wire
from src.domain.image_geometry import letterbox_geometry, whole_square_geometry


if TYPE_CHECKING:
    from collections.abc import Iterable


EMBED_INPUT = 'images'
EMBED_OUTPUT = 'image_embeddings'


def _square(spec: wire.EmbedSpec | wire.CropEmbedSpec, n: int) -> ModelInput:
    return ModelInput(EMBED_INPUT, 'UINT8', (n, 3, spec.size, spec.size))


def _vectors(photo: Photo, calls: Iterable[Call]) -> list[np.ndarray]:
    return [photo.calls.result(c)[EMBED_OUTPUT].astype(np.float32) for c in calls]


class EmbedFrame(Branch):
    status = 'embed_frame'

    def __init__(self, spec: wire.EmbedSpec) -> None:
        self.spec = spec
        self.call: Call | None = None

    def round1(self, photo: Photo) -> None:
        t = self.spec.size
        w, h = photo.size
        tensor = _square(self.spec, 1)
        staged = photo.calls.stage(tensor.nbytes, 'embed_frame')
        with photo.clock.stage('embed_frame'):
            g = whole_square_geometry(w, h, t)
            square = staged.device.reshape(tensor.shape)[0]
            photo.kernels.resample_into(
                photo.frame, (0, 0, w, h), (g.new_w, g.new_h), square, (g.left, g.top, t, t)
            )
        self.call = photo.calls.submit(
            self, 'embed_pe', staged, self.spec.model, tensor, (EMBED_OUTPUT,)
        )

    def finish(self, photo: Photo) -> None:
        if self.call is None:
            raise RuntimeError('embed_frame finished before round 1')
        [vector] = _vectors(photo, [self.call])
        photo.out[wire.WHOLE_EMBEDDING] = vector[0]


class EmbedCrops(Branch):
    status = 'embed_crops'

    def __init__(self, spec: wire.CropEmbedSpec, classes: frozenset[int] | None) -> None:
        self.spec = spec
        self.classes = classes
        self.mask: np.ndarray | None = None
        self.calls: list[Call] = []

    def round2(self, photo: Photo) -> None:
        if photo.rects is None:
            raise DependencyFailedError('no rectangles')
        mask = np.ones(len(photo.rects), np.bool_)
        if self.classes is not None and photo.classes is not None:
            mask = np.isin(photo.classes, np.array(sorted(self.classes), np.int32))
        self.mask = mask
        picked = [r for r, keep in zip(photo.rects, mask, strict=True) if keep]
        t = self.spec.size
        for s in range(0, len(picked), self.spec.batch):
            chunk = picked[s : s + self.spec.batch]
            tensor = _square(self.spec, len(chunk))
            staged = photo.calls.stage(tensor.nbytes, 'embed_crops')
            with photo.clock.stage('embed_crops'):
                squares = staged.device.reshape(tensor.shape)
                squares.fill(self.spec.fill)
                for k, (x1, y1, x2, y2) in enumerate(chunk):
                    g = letterbox_geometry(x2 - x1, y2 - y1, t)
                    photo.kernels.resample_into(
                        photo.frame,
                        (x1, y1, x2 - x1, y2 - y1),
                        (g.new_w, g.new_h),
                        squares[k, :, g.top : g.top + g.new_h, g.left : g.left + g.new_w],
                        antialias=True,
                    )
            self.calls.append(
                photo.calls.submit(
                    self, 'embed_pe', staged, self.spec.model, tensor, (EMBED_OUTPUT,)
                )
            )

    def finish(self, photo: Photo) -> None:
        if self.mask is None:
            raise RuntimeError('embed_crops finished before round 2')
        rows = _vectors(photo, self.calls)
        # nothing picked: no row tells the crop encoder's width, and it need not be the frame's
        vectors = np.concatenate(rows, axis=0) if rows else np.zeros((0, 0), np.float32)
        photo.out[wire.CROP_EMBEDDINGS] = vectors  # [picked, dim]
        photo.out[wire.CROP_EMBEDDED] = self.mask
