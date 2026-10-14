"""Torch kernels of the GPU decode spike (#217): the CPU preprocessing of ingest, re-expressed on
a decoded uint8 frame tensor that can stay on the GPU.

Each function names the product function it stands in for and how close it is:

* ``letterbox_u8``  ``geometry.letterbox_to_square``: identical geometry, pad value and box
  un-map; Pillow's antialiased bilinear resampling is reproduced by ``F.interpolate(...,
  antialias=True)`` in float (Pillow rounds in fixed point), so pixels differ by about one
  grey level (gate T).
* ``pe_square_u8``  ``pe_preprocess.resize_crop_rgb``: identical geometry; OpenCV's
  ``INTER_LINEAR`` (fixed point, no antialiasing) by bilinear without antialiasing (gate T).
* ``crop_u8``       ``ingest_index.crop_pil``: bit-identical, zeros outside the frame.
* ``gray_u8``, ``lap_var``  ``crop_quality.image_lap_var`` / ``crop_lap_var``: bit-identical
  (OpenCV's fixed-point grey weights, the integer Laplacian, the integer variance formula).
* ``whole_frame_u8``  ``pe_preprocess.whole_frame_rgb``: **not** the same pixels. The product
  decodes the whole frame a second time at 1/8 scale in the DCT domain (OpenCV
  ``IMREAD_REDUCED_COLOR_8``); a GPU path that decodes once resizes the full frame instead.

Device-agnostic: the unit tests run them on CPU torch, the probes on CUDA tensors.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812  # conventional alias

from scripts.bench import gpu_decode_lib as lib


def _to_u8(x: torch.Tensor) -> torch.Tensor:
    return x.round().clamp_(0, 255).to(torch.uint8)


def _nchw_float(hwc: torch.Tensor) -> torch.Tensor:
    return hwc.permute(2, 0, 1)[None].float()


def letterbox_u8(
    frame: torch.Tensor, target: int = 640, fill: int = 114
) -> tuple[torch.Tensor, float, tuple[float, float]]:
    """HWC uint8 frame -> ``(1, 3, target, target)`` uint8, scale, ``(pad_w, pad_h)``."""
    h, w = int(frame.shape[0]), int(frame.shape[1])
    g = lib.letterbox_geometry(w, h, target)
    resized = F.interpolate(
        _nchw_float(frame), size=(g.new_h, g.new_w), mode='bilinear', antialias=True
    )
    canvas = torch.full((1, 3, target, target), fill, dtype=torch.uint8, device=frame.device)
    canvas[:, :, g.top : g.top + g.new_h, g.left : g.left + g.new_w] = _to_u8(resized)
    return canvas, g.scale, (g.pad_w, g.pad_h)


def pe_square_u8(rgb: torch.Tensor, target: int = 336) -> torch.Tensor:
    """HWC uint8 -> ``(target, target, 3)`` uint8: shorter edge to ``target``, centre crop."""
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if h == 0 or w == 0:
        return torch.zeros((target, target, 3), dtype=torch.uint8, device=rgb.device)
    g = lib.pe_square_geometry(w, h, target)
    resized = F.interpolate(
        _nchw_float(rgb), size=(g.new_h, g.new_w), mode='bilinear', align_corners=False
    )
    out = resized[0, :, g.top : g.top + target, g.left : g.left + target]
    return _to_u8(out).permute(1, 2, 0).contiguous()


def crop_u8(frame: torch.Tensor, rect: tuple[int, int, int, int]) -> torch.Tensor:
    """The pixels of ``rect`` (``x1, y1, x2, y2``, x2/y2 exclusive), zeros outside the frame."""
    x1, y1, x2, y2 = rect
    h, w = int(frame.shape[0]), int(frame.shape[1])
    out = torch.zeros((y2 - y1, x2 - x1, 3), dtype=torch.uint8, device=frame.device)
    sx2, sy2 = min(x2, w), min(y2, h)
    if sx2 > x1 and sy2 > y1:
        out[: sy2 - y1, : sx2 - x1] = frame[y1:sy2, x1:sx2]
    return out


def gray_u8(rgb: torch.Tensor) -> torch.Tensor:
    """OpenCV ``COLOR_RGB2GRAY`` for uint8: fixed-point weights, 15 fractional bits, rounded
    (checked bit for bit against OpenCV 4.10 of the API image and 5.0)."""
    t = rgb.to(torch.int32)
    return ((t[..., 0] * 9798 + t[..., 1] * 19235 + t[..., 2] * 3735 + 16384) >> 15).to(torch.uint8)


def _reflect101(n: int, device: Any) -> torch.Tensor:
    if n == 1:
        return torch.zeros(3, dtype=torch.long, device=device)
    idx = torch.arange(-1, n + 1, device=device)
    idx[0], idx[-1] = 1, n - 2
    return idx


def lap_sums_gray(gray: torch.Tensor, rows_per_chunk: int = 0) -> tuple[int, int, int]:
    """``(n, sum, sum of squares)`` of OpenCV's 3x3 Laplacian (``BORDER_REFLECT_101``) of a
    uint8 grey plane, in exact integers. ``rows_per_chunk`` bounds the int64 temporaries (an
    18 MP plane is 144 MB per int64 copy); the sums do not depend on it."""
    h, w = int(gray.shape[0]), int(gray.shape[1])
    rows = _reflect101(h, gray.device)
    cols = _reflect101(w, gray.device)
    step = rows_per_chunk if rows_per_chunk > 0 else h
    total = squares = 0
    for r0 in range(0, h, step):
        r1 = min(h, r0 + step)
        p = gray.index_select(0, rows[r0 : r1 + 2]).index_select(1, cols).to(torch.int32)
        lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * p[1:-1, 1:-1]
        lap64 = lap.to(torch.int64)
        total += int(lap64.sum())
        squares += int((lap64 * lap64).sum())
    return h * w, total, squares


def lap_sums(rgb: torch.Tensor, rows_per_chunk: int = 0) -> tuple[int, int, int]:
    """:func:`lap_sums_gray` of the grey plane of an RGB frame."""
    return lap_sums_gray(gray_u8(rgb), rows_per_chunk)


def var_from_sums(n: int, total: int, squares: int) -> float:
    """``crop_quality.laplacian_var``'s integer variance formula, rounded to 2 decimals."""
    return round((n * squares - total * total) / (n * n), 2)


def lap_var(rgb: torch.Tensor) -> float:
    """``crop_quality.laplacian_var`` of the grey plane of ``rgb``, rounded to 2 decimals."""
    return var_from_sums(*lap_sums(rgb))


def whole_frame_u8(frame: torch.Tensor, target: int = 336, mode: str = 'antialias') -> torch.Tensor:
    """The PE whole-frame square from the full decoded frame (one decode for everything).

    ``antialias``: antialiased bilinear straight to the shorter-edge size; ``area``: box
    average (``F.interpolate`` ``area``); ``dct8``: emulates the product's 1/8 DCT decode (an
    8 x 8 block mean is what a 1/8-scaled IDCT keeps, the DC term), then OpenCV's
    ``INTER_LINEAR`` from that thumbnail, as ``whole_frame_rgb`` does.
    """
    if mode == 'dct8':
        thumb = F.avg_pool2d(_nchw_float(frame), 8, stride=8, ceil_mode=True)
        thumb_hwc = _to_u8(thumb[0]).permute(1, 2, 0)
        return pe_square_u8(thumb_hwc, target)
    if mode not in ('antialias', 'area'):
        msg = f'unknown mode {mode!r}'
        raise ValueError(msg)
    h, w = int(frame.shape[0]), int(frame.shape[1])
    g = lib.pe_square_geometry(w, h, target)
    x = _nchw_float(frame)
    if mode == 'antialias':
        r = F.interpolate(x, size=(g.new_h, g.new_w), mode='bilinear', antialias=True)
    else:
        r = F.interpolate(x, size=(g.new_h, g.new_w), mode='area')
    out = r[0, :, g.top : g.top + target, g.left : g.left + target]
    return _to_u8(out).permute(1, 2, 0).contiguous()


def apply_orientation(frame: torch.Tensor, orientation: int) -> torch.Tensor:
    """Re-orient an HWC frame by its EXIF tag as ``ImageOps.exif_transpose`` does (for
    decoders that ignore the tag)."""
    t = frame
    if orientation in (5, 6, 7, 8):
        t = t.permute(1, 0, 2)
    flips = {2: [1], 3: [0, 1], 4: [0], 6: [1], 7: [0, 1], 8: [0]}.get(orientation)
    if flips:
        t = t.flip(flips)
    return t.contiguous()
