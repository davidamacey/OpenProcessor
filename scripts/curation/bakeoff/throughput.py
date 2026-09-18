"""Steady-state inference throughput benchmark for the LPR model variants.

The bake-off ``run.py`` reports per-image latency that includes disk I/O
(``cv2.imread``), which dominates and masks the model's real speed. This tool
isolates inference: it pre-loads a fixed set of frames into memory once, warms
up, then loops ``detect()`` over them at steady state and reports images/sec.

Run the SAME tool per backend/EP to get a fair CPU-vs-GPU speedup table:

    # GPU (CUDA EP)
    python -m scripts.curation.bakeoff.throughput --backend onnxruntime \
        --weights /data/quant/ours_yolo26n/fp16.onnx --label fp16-cuda \
        --ort-providers CUDAExecutionProvider,CPUExecutionProvider \
        --images /data/lpr_exports/<run>/images/test --out /data/quant/throughput

    # CPU EP
    python -m scripts.curation.bakeoff.throughput --backend onnxruntime \
        --weights /data/quant/ours_yolo26n/int8_qdq.onnx --label int8-cpu \
        --ort-providers CPUExecutionProvider --images ... --out ...

    # Apple CoreML (macOS)
    python -m scripts.curation.bakeoff.throughput --backend coreml \
        --weights ~/lpr_quant/quant/ours_yolo26n/int8.mlpackage --label int8-ane ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _build_detector(args: argparse.Namespace) -> Any:
    """Construct the model-under-test (same backends as run.py)."""
    if args.backend == 'ultralytics':
        from .backends.ultralytics_pt import UltralyticsDetector

        return UltralyticsDetector(
            args.weights,
            name=args.label,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
        )
    if args.backend == 'onnxruntime':
        from .backends.onnxruntime_ort import OnnxRuntimeDetector

        return OnnxRuntimeDetector(
            args.weights,
            name=args.label,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            providers=args.ort_providers,
            coords_normalized=args.coords_normalized,
        )
    if args.backend == 'coreml':
        from .backends.coreml import CoreMLDetector

        return CoreMLDetector(
            args.weights,
            name=args.label,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            compute_units=args.coreml_compute_units,
            coords_normalized=args.coords_normalized,
        )
    if args.backend == 'triton':
        from .backends.triton_trt import TritonLprDetector

        return TritonLprDetector(
            url=args.triton_url,
            model=args.triton_model,
            input_size=args.imgsz,
            conf=args.conf,
            iou=args.iou,
        )
    raise SystemExit(f'unknown backend {args.backend!r}')


def _load_images(images_dir: Path, n: int) -> list[np.ndarray]:
    """Decode up to ``n`` frames into memory once (RGB), so timing excludes I/O."""
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)[:n]
    frames: list[np.ndarray] = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is not None:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not frames:
        raise SystemExit(f'no images decoded under {images_dir}')
    return frames


def _percentiles(latencies_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        'mean': float(arr.mean()),
        'p50': float(np.percentile(arr, 50)),
        'p90': float(np.percentile(arr, 90)),
        'p99': float(np.percentile(arr, 99)),
        'min': float(arr.min()),
        'max': float(arr.max()),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        '--backend', required=True, choices=['ultralytics', 'onnxruntime', 'coreml', 'triton']
    )
    p.add_argument('--weights')
    p.add_argument('--label', required=True, help='Name for this variant/EP in the report')
    p.add_argument('--images', type=Path, required=True, help='Dir of frames to loop over')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='0')
    p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--ort-providers', default='CUDAExecutionProvider,CPUExecutionProvider')
    p.add_argument('--coreml-compute-units', default='ALL')
    p.add_argument('--coords-normalized', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--triton-url', default='localhost:4601')
    p.add_argument('--triton-model', default='lpr_nanov11_640')
    p.add_argument('--n-images', type=int, default=200, help='Frames pre-loaded into memory')
    p.add_argument('--warmup', type=int, default=30, help='Untimed warmup detect() calls')
    p.add_argument('--min-seconds', type=float, default=8.0, help='Minimum steady-state duration')
    p.add_argument('--out', type=Path, help='Dir to write <label>.throughput.json')
    args = p.parse_args()

    frames = _load_images(args.images, args.n_images)
    det = _build_detector(args)
    runtime = getattr(det, 'runtime', args.backend)
    print(f'throughput: {args.label} [{runtime}] over {len(frames)} in-memory frames', flush=True)

    # Warmup (JIT / engine build / cuDNN autotune / ANE spin-up).
    for i in range(args.warmup):
        det.detect(frames[i % len(frames)])

    # Steady state: loop until both min-seconds elapsed and >= n_images done.
    latencies_ms: list[float] = []
    count = 0
    t_start = time.perf_counter()
    i = 0
    while True:
        frame = frames[i % len(frames)]
        t0 = time.perf_counter()
        det.detect(frame)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        count += 1
        i += 1
        elapsed = time.perf_counter() - t_start
        if elapsed >= args.min_seconds and count >= args.n_images:
            break
    total_s = time.perf_counter() - t_start

    lat = _percentiles(latencies_ms)
    fps = count / total_s if total_s > 0 else 0.0
    report: dict[str, Any] = {
        'label': args.label,
        'backend': args.backend,
        'runtime': runtime,
        'weights': args.weights,
        'imgsz': args.imgsz,
        'images_processed': count,
        'duration_s': round(total_s, 3),
        'throughput_fps': round(fps, 2),
        'latency_ms': {k: round(v, 3) for k, v in lat.items()},
    }
    print(
        f'  {args.label}: {fps:.1f} img/s  (mean {lat["mean"]:.2f} ms, p50 {lat["p50"]:.2f}, '
        f'p90 {lat["p90"]:.2f})  over {count} frames in {total_s:.1f}s',
        flush=True,
    )
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        safe = args.label.replace('/', '_').replace(' ', '_')
        (args.out / f'{safe}.throughput.json').write_text(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
