#!/usr/bin/env python3
"""What GPU decode and GPU resizing do to ingest's numbers (#217), measured against the
product's CPU path on the same images, through the same detector and PE engines.

``capture`` runs every image three ways and stores what ingest would store:

* ``cpu``  the product: Pillow decode + EXIF (``ingest._decode_image``), Pillow letterbox,
           crops by ``crop_pil`` + ``resize_crop_rgb``, the whole frame from OpenCV's 1/8
           DCT decode, OpenCV blur.
* ``dec``  GPU decode (``--decoder``: nvImageCodec, default, or torchvision), then the product's CPU functions on the
           decoded pixels; the whole frame still from OpenCV's 1/8 decode. Isolates the decoder.
* ``gpu``  nvJPEG decode and every derived tensor on the GPU (``gpu_decode_kernels``): one
           decode, letterbox, crops, PE squares, whole frame (antialiased resize of the full
           frame; ``area`` recorded too), blur.

Cascade probes (owner question: resize each model input from the next larger one): the PE
whole frame and the PE crops cut from the detector's 640 letterbox instead of the full frame.

A file nvJPEG rejects is decoded by the CPU for ``dec`` and ``gpu`` and counted in
``fallbacks``. ``compare`` turns a capture into the gate table (``P-box``, ``P-item``,
``P-pe``, ``P-pew``, kNN, ``P-blur``, gate T) and writes the detections of each arm in the
``detector_quality.py`` format. Needs the API image (product Pillow/OpenCV, torch with CUDA,
tritonclient) and a Triton with the detector and PE engines.

    python scripts/bench/gpu_decode_parity.py capture --images DIR --label A --url 127.0.0.1:4911 \\
        --out /out/par_a.npz
    python scripts/bench/gpu_decode_parity.py compare --capture /out/par_a.npz --out /out/par_a.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.bench import gpu_decode_lib as lib, parity_metrics as pm  # noqa: E402  # import follows the sys.path setup above
from src.config.settings import TritonModelConfig  # noqa: E402  # import follows the sys.path setup above


DetRow = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
ARMS = ('cpu', 'cpu_b1', 'dec', 'gpu')
DET_MODEL = TritonModelConfig.YOLO_MODEL
PE_MODEL = TritonModelConfig.PE_IMAGE_MODEL
DET_OUTPUTS = ('num_dets', 'det_boxes', 'det_scores', 'det_classes')


class Engines:
    def __init__(self, url: str) -> None:
        import tritonclient.grpc as grpcclient

        self._g = grpcclient
        self.client = grpcclient.InferenceServerClient(url)

    def _infer(self, model: str, x: np.ndarray, outputs: tuple[str, ...]) -> list[np.ndarray]:
        inp = self._g.InferInput('images', list(x.shape), 'UINT8')
        inp.set_data_from_numpy(np.ascontiguousarray(x))
        res = self.client.infer(
            model, [inp], outputs=[self._g.InferRequestedOutput(o) for o in outputs]
        )
        return [res.as_numpy(o) for o in outputs]

    def detect(self, chws: list[np.ndarray]) -> list[DetRow]:
        rows: list[DetRow] = []
        for i in range(0, len(chws), 16):
            outs = self._infer(DET_MODEL, np.stack(chws[i : i + 16]), DET_OUTPUTS)
            rows.extend(
                (outs[0][j], outs[1][j], outs[2][j], outs[3][j]) for j in range(len(outs[0]))
            )
        return rows

    def embed(self, squares_hwc: list[np.ndarray]) -> np.ndarray:
        if not squares_hwc:
            return np.zeros((0, 1024), np.float32)
        out = []
        for i in range(0, len(squares_hwc), 32):
            x = np.stack(squares_hwc[i : i + 32]).transpose(0, 3, 1, 2)
            out.append(self._infer(PE_MODEL, x, ('image_embeddings',))[0])
        return np.concatenate(out)


_DECODER: dict[str, Any] = {}


def _gpu_decode(data: bytes) -> Any:
    """The configured GPU decoder (``_DECODER['name']``) on one JPEG, as an HWC uint8 CUDA tensor.

    ``nvimgcodec``: nvImageCodec with libjpeg-style (fancy) chroma upsampling, EXIF applied by
    the decoder. ``torchvision``: nvJPEG through torchvision; its CUDA decode ignores the EXIF
    orientation (measured with ``gpu_decode_bench.py formats``), so the frame is re-oriented
    from the header.
    """
    import torch

    from scripts.bench import gpu_decode_kernels as k

    if _DECODER['name'] == 'nvimgcodec':
        if 'dec' not in _DECODER:
            from nvidia import nvimgcodec

            _DECODER['dec'] = nvimgcodec.Decoder(device_id=0, options=':fancy_upsampling=1')
        img = _DECODER['dec'].decode(data)
        if img is None:
            msg = 'nvimgcodec returned no image'
            raise RuntimeError(msg)
        return torch.as_tensor(img, device='cuda').clone()
    import torchvision

    t = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    chw = torchvision.io.decode_jpeg(t, device='cuda', mode=torchvision.io.ImageReadMode.RGB)
    return k.apply_orientation(chw.permute(1, 2, 0), lib.jpeg_header(data).orientation)


def _dets(
    row: DetRow,
    scale: float,
    pad: tuple[float, float],
) -> list[list[float]]:
    from scripts.bench.detector_capture import decode_row

    return decode_row(*row, scale, pad)


def _box4(box: list[float]) -> tuple[float, float, float, float]:
    return (box[0], box[1], box[2], box[3])


def _clamped_rect(box: list[float], w: int, h: int) -> tuple[int, int, int, int] | None:
    """``crop_lap_var``'s rectangle: rounded, clamped to the frame, None when empty."""
    x1, y1, x2, y2 = (round(v) for v in box[:4])
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    return None if x2 <= x1 or y2 <= y1 else (x1, y1, x2, y2)


def capture_image(name: str, data: bytes, eng: Engines, out: dict[str, Any]) -> None:
    import torch
    from PIL import Image

    from scripts.bench import gpu_decode_kernels as k
    from src.services.curation.ingest import _decode_image
    from src.services.curation.ingest_index import crop_pil
    from src.services.detection.crop_quality import crop_lap_var, image_lap_var
    from src.services.detection.geometry import letterbox_to_square
    from src.services.detection.pe_preprocess import resize_crop_rgb, whole_frame_rgb_from_bytes

    pil = _decode_image(data).pil
    w, h = pil.size
    rgb = np.asarray(pil)
    try:
        frame = _gpu_decode(data)
        fallback = None
    except Exception as exc:
        frame, fallback = torch.from_numpy(rgb.copy()).cuda(), f'{type(exc).__name__}: {exc}'[:120]
        out['fallbacks'].append(
            {'name': name, 'error': fallback, **lib.jpeg_header(data)._asdict()}
        )
    dec_rgb = frame.cpu().numpy()
    if dec_rgb.shape != rgb.shape:
        out['fallbacks'].append({'name': name, 'error': f'shape {dec_rgb.shape} vs {rgb.shape}'})
        frame, dec_rgb = torch.from_numpy(rgb.copy()).cuda(), rgb
    dec_pil = Image.fromarray(dec_rgb)
    d = np.abs(dec_rgb.astype(np.int16) - rgb.astype(np.int16))
    out['decode_diff'].append((name, float(d.mean()), int(d.max()), float((d > 0).mean())))

    cpu_lb, scale, pad = letterbox_to_square(pil, 640)
    dec_lb, _, _ = letterbox_to_square(dec_pil, 640)
    gpu_lb_t, g_scale, g_pad = k.letterbox_u8(frame, 640)
    assert (g_scale, g_pad) == (scale, pad)
    gpu_lb = gpu_lb_t.cpu().numpy()
    lbs = {'cpu': cpu_lb[0], 'dec': dec_lb[0], 'gpu': gpu_lb[0]}
    for arm in ('dec', 'gpu'):
        t = lib.tensor_diff(lbs['cpu'].transpose(1, 2, 0), lbs[arm].transpose(1, 2, 0))
        out['t_letterbox'][arm].append((name, *t.best_shift, t.mean_abs, t.max_abs))
    rows = eng.detect([lbs[a] for a in ('cpu', 'dec', 'gpu')])
    rows.extend(eng.detect([lbs['cpu']]))
    arm_rows = dict(zip(('cpu', 'dec', 'gpu', 'cpu_b1'), rows, strict=True))
    dets = {a: _dets(arm_rows[a], scale, pad) for a in ARMS}

    cpu_whole = whole_frame_rgb_from_bytes(data)
    assert cpu_whole is not None
    gpu_whole = k.whole_frame_u8(frame).cpu().numpy()
    area_whole = k.whole_frame_u8(frame, mode='area').cpu().numpy()
    dct8_whole = k.whole_frame_u8(frame, mode='dct8').cpu().numpy()
    g = lib.letterbox_geometry(w, h, 640)
    inner = gpu_lb_t[0, :, g.top : g.top + g.new_h, g.left : g.left + g.new_w].permute(1, 2, 0)
    cascade_whole = k.whole_frame_u8(inner).cpu().numpy()
    lb_hwc = gpu_lb_t[0].permute(1, 2, 0)
    t8 = lib.tensor_diff(cpu_whole, dct8_whole)
    out['t_whole_dct8'].append((name, *t8.best_shift, t8.mean_abs, t8.max_abs))
    t = lib.tensor_diff(cpu_whole, gpu_whole)
    out['t_whole'].append((name, *t.best_shift, t.mean_abs, t.max_abs))
    cpu_var = image_lap_var(rgb, 'rgb')
    full_var = {
        'cpu': cpu_var,
        'cpu_b1': cpu_var,
        'dec': image_lap_var(dec_rgb, 'rgb'),
        'gpu': k.lap_var(frame),
    }

    squares: dict[str, list[np.ndarray]] = {
        a: [] for a in (*ARMS, 'gpu_on_cpu_boxes', 'cascade_on_cpu_boxes')
    }
    for arm in ARMS:
        src_pil = pil if arm.startswith('cpu') else dec_pil
        src_rgb = rgb if arm.startswith('cpu') else dec_rgb
        for det in dets[arm]:
            box = det[:4]
            if arm == 'gpu':
                sq = k.pe_square_u8(k.crop_u8(frame, lib.crop_rect(_box4(box)))).cpu().numpy()
                r = _clamped_rect(box, w, h)
                bv = None if r is None else k.lap_var(frame[r[1] : r[3], r[0] : r[2]])
            else:
                sq = resize_crop_rgb(np.asarray(crop_pil(src_pil, _box4(box))))
                bv = crop_lap_var(src_rgb, _box4(box), 'rgb')
            squares[arm].append(sq)
            out['items'][arm].append((name, *det, bv, full_var[arm], *lib.crop_rect(_box4(box))))
    for det in dets['cpu']:
        sq = k.pe_square_u8(k.crop_u8(frame, lib.crop_rect(_box4(det)))).cpu().numpy()
        squares['gpu_on_cpu_boxes'].append(sq)
        lb_box = _box4(
            [v * scale + (pad[0] if i % 2 == 0 else pad[1]) for i, v in enumerate(det[:4])]
        )
        squares['cascade_on_cpu_boxes'].append(
            k.pe_square_u8(k.crop_u8(lb_hwc, lib.crop_rect(lb_box))).cpu().numpy()
        )
        cpu_sq = squares['cpu'][len(squares['gpu_on_cpu_boxes']) - 1]
        t = lib.tensor_diff(cpu_sq, sq)
        out['t_crop'].append((name, *t.best_shift, t.mean_abs, t.max_abs))
    frames = [cpu_whole, gpu_whole, area_whole, dct8_whole, cascade_whole]
    vec = eng.embed([*frames, *(s for a in squares for s in squares[a])])
    out['frame_vec'].append((name, vec[:5].astype(np.float16)))
    offset = 5
    for arm, arm_squares in squares.items():
        n = len(arm_squares)
        out['item_vec'][arm].append(vec[offset : offset + n].astype(np.float16))
        offset += n


def run_capture(args: argparse.Namespace) -> int:
    _DECODER['name'] = args.decoder
    eng = Engines(args.url)
    names = sorted(p for p in Path(args.images).iterdir() if p.is_file())[: args.limit or None]
    out: dict[str, Any] = {
        'fallbacks': [],
        'decode_diff': [],
        't_letterbox': {'dec': [], 'gpu': []},
        't_whole': [],
        't_whole_dct8': [],
        't_crop': [],
        'items': {a: [] for a in ARMS},
        'item_vec': {a: [] for a in (*ARMS, 'gpu_on_cpu_boxes', 'cascade_on_cpu_boxes')},
        'frame_vec': [],
    }
    for i, p in enumerate(names):
        capture_image(p.name, p.read_bytes(), eng, out)
        if i % 100 == 0:
            print(f'{i}/{len(names)}', flush=True)
    arrays: dict[str, np.ndarray] = {
        'frame_image': np.array([n for n, _ in out['frame_vec']]),
        'frame_vec': np.stack([v for _, v in out['frame_vec']]),
        'decode_diff': np.array([r[1:] for r in out['decode_diff']], dtype=np.float64),
        't_whole': np.array([r[1:] for r in out['t_whole']], dtype=np.float64),
        't_whole_dct8': np.array([r[1:] for r in out['t_whole_dct8']], dtype=np.float64),
        't_crop': np.array([r[1:] for r in out['t_crop']], dtype=np.float64).reshape(-1, 4),
    }
    for arm in ('dec', 'gpu'):
        arrays[f't_letterbox_{arm}'] = np.array(
            [r[1:] for r in out['t_letterbox'][arm]], dtype=np.float64
        )
    for arm in ARMS:
        rows = out['items'][arm]
        arrays[f'{arm}_image'] = np.array([r[0] for r in rows])
        arrays[f'{arm}_det'] = np.array([r[1:7] for r in rows], dtype=np.float64).reshape(-1, 6)
        arrays[f'{arm}_blur'] = np.array(
            [[np.nan if r[7] is None else r[7], r[8]] for r in rows], dtype=np.float64
        ).reshape(-1, 2)
        arrays[f'{arm}_rect'] = np.array([r[9:13] for r in rows], dtype=np.int32).reshape(-1, 4)
    for arm, chunks in out['item_vec'].items():
        arrays[f'{arm}_vec'] = np.concatenate(chunks) if chunks else np.zeros((0, 1024), np.float16)
    np.savez_compressed(args.out, **arrays)  # type: ignore[arg-type]  # mypy reads **dict as the allow_pickle flag
    Path(args.out).with_suffix('.fallbacks.json').write_text(json.dumps(out['fallbacks'], indent=1))
    print(json.dumps({'images': len(names), 'fallbacks': len(out['fallbacks'])}))
    return 0


# ------------------------------------------------------------------------------- compare


def _boxes(a: dict[str, np.ndarray], arm: str) -> DetRow:
    det = a[f'{arm}_det']
    return a[f'{arm}_image'], det[:, :4], det[:, 5].astype(int).astype(str), det[:, 4]


def _pairs(a: dict[str, np.ndarray], arm: str) -> list[tuple[int, int]]:
    ref, cand = _boxes(a, 'cpu'), _boxes(a, arm)
    pairs: list[tuple[int, int]] = []
    for name in np.unique(ref[0]):
        ri = np.flatnonzero(ref[0] == name)
        ci = np.flatnonzero(cand[0] == name)
        got, _, _ = pm.match_boxes(ref[1][ri], ref[2][ri], cand[1][ci], cand[2][ci])
        pairs.extend((int(ri[r]), int(ci[c])) for r, c in got)
    return pairs


def _t_share(t: np.ndarray) -> dict[str, Any]:
    if len(t) == 0:
        return {'n': 0}
    ok = (t[:, 0] == 0) & (t[:, 1] == 0) & (t[:, 2] < 0.5) & (t[:, 3] <= 8)
    return {
        'n': len(t),
        'pass_share': float(ok.mean()),
        'shift_zero_share': float(((t[:, 0] == 0) & (t[:, 1] == 0)).mean()),
        'mean_abs': lib.summarize(t[:, 2].tolist()),
        'max_abs': lib.summarize(t[:, 3].tolist()),
    }


def compare(a: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {
        'images': len(a['frame_image']),
        'decode_mean_abs': lib.summarize(a['decode_diff'][:, 0].tolist()),
        'decode_max_abs': lib.summarize(a['decode_diff'][:, 1].tolist()),
        'decode_share_differing': lib.summarize(a['decode_diff'][:, 2].tolist()),
        'T_letterbox': {arm: _t_share(a[f't_letterbox_{arm}']) for arm in ('dec', 'gpu')},
        'T_whole_frame': _t_share(a['t_whole']),
        'T_whole_frame_dct8': _t_share(a['t_whole_dct8']),
        'T_crop_square': _t_share(a['t_crop']),
    }
    fv = a['frame_vec'].astype(np.float32)
    out['P-pew'] = {
        'gpu_antialias': pm.cosine_summary(pm.cosine_rows(fv[:, 0], fv[:, 1])),
        'gpu_area': pm.cosine_summary(pm.cosine_rows(fv[:, 0], fv[:, 2])),
        'gpu_dct8': pm.cosine_summary(pm.cosine_rows(fv[:, 0], fv[:, 3])),
        'cascade_from_letterbox': pm.cosine_summary(pm.cosine_rows(fv[:, 0], fv[:, 4])),
        'cascade_vs_full_antialias': pm.cosine_summary(pm.cosine_rows(fv[:, 1], fv[:, 4])),
    }
    cpu_vec = a['cpu_vec'].astype(np.float32)
    same_boxes = pm.cosine_rows(cpu_vec, a['gpu_on_cpu_boxes_vec'].astype(np.float32))
    out['P-pe_same_boxes_gpu_resize'] = pm.cosine_summary(same_boxes)
    cascade = pm.cosine_rows(cpu_vec, a['cascade_on_cpu_boxes_vec'].astype(np.float32))
    out['P-pe_same_boxes_cascade_from_letterbox'] = pm.cosine_summary(cascade)
    for arm in ('cpu_b1', 'dec', 'gpu'):
        res: dict[str, Any] = pm.box_metrics(_boxes(a, 'cpu'), _boxes(a, arm))
        pairs = _pairs(a, arm)
        ri = np.array([p[0] for p in pairs], dtype=int)
        ci = np.array([p[1] for p in pairs], dtype=int)
        rect_same = (a['cpu_rect'][ri] == a[f'{arm}_rect'][ci]).all(axis=1)
        res['crop_rect_identical_share'] = float(rect_same.mean()) if len(ri) else None
        cand_vec = a[f'{arm}_vec'].astype(np.float32)
        res['P-pe'] = pm.cosine_summary(pm.cosine_rows(cpu_vec[ri], cand_vec[ci]))
        res['knn'] = pm.knn_agreement(
            cpu_vec[ri], cand_vec[ci], k=10, n_queries=1000, seed=0, dup_threshold=0.98
        )
        cb, gb = a['cpu_blur'][ri], a[f'{arm}_blur'][ci]
        ok = ~np.isnan(cb[:, 0]) & ~np.isnan(gb[:, 0]) & (cb[:, 1] > 1e-6) & (gb[:, 1] > 1e-6)
        ratio_c = np.round(cb[ok, 0] / cb[ok, 1], 2)
        ratio_g = np.round(gb[ok, 0] / gb[ok, 1], 2)
        res['P-blur_ratio_abs_diff'] = lib.summarize(np.abs(ratio_c - ratio_g).tolist())
        res['P-blur_pass_share'] = (
            float((np.abs(ratio_c - ratio_g) <= 0.02).mean()) if ok.any() else None
        )
        res['full_var_rel_diff'] = lib.summarize(
            (np.abs(gb[ok, 1] - cb[ok, 1]) / cb[ok, 1]).tolist()
        )
        out[arm] = res
    return out


def to_dets(a: dict[str, np.ndarray], arm: str) -> dict[str, list[list[float]]]:
    """``detector_quality.py`` input: ``{coco image id: [[x1, y1, x2, y2, score, class], ...]}``."""
    out: dict[str, list[list[float]]] = {str(int(Path(n).stem)): [] for n in a['frame_image']}
    for name, det in zip(a[f'{arm}_image'], a[f'{arm}_det'], strict=True):
        out[str(int(Path(str(name)).stem))].append([*map(float, det[:5]), int(det[5])])
    return out


def run_compare(args: argparse.Namespace) -> int:
    with np.load(args.capture) as z:
        a = {k: z[k] for k in z.files}
    report = compare(a)
    Path(args.out).write_text(json.dumps(report, indent=1, default=float))
    if args.dets_dir:
        for arm in ARMS:
            Path(args.dets_dir, f'dets_{arm}.json').write_text(json.dumps(to_dets(a, arm)))
    print(json.dumps(report, indent=1, default=float)[:4000])
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--images', required=True)
    c.add_argument('--url', required=True, help='Triton gRPC host:port')
    c.add_argument('--limit', type=int, default=0)
    c.add_argument('--decoder', choices=('nvimgcodec', 'torchvision'), default='nvimgcodec')
    c.add_argument('--out', required=True, help='.npz path')
    d = sub.add_parser('compare')
    d.add_argument('--capture', required=True)
    d.add_argument('--out', required=True)
    d.add_argument('--dets-dir', help='write dets_<arm>.json for detector_quality.py (COCO names)')
    args = p.parse_args(argv)
    return run_capture(args) if args.cmd == 'capture' else run_compare(args)


if __name__ == '__main__':
    raise SystemExit(main())
