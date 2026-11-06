#!/usr/bin/env python3
"""gpu:PR-0: the harness and the honest GPU number.

Runs the shipped ``gpu_image_prep`` algorithm directly (no Triton) on one image set and writes, per
repetition, throughput, per-stage wall time, NVML samples and, with ``--check``, the parity of the
GPU letterbox, whole-frame square and crop squares against the Pillow/OpenCV references.

    # tier 0, CPU only (no GPU is touched)
    pr0_stage_split.py cpu-smoke --root DIR [--pin FILE | --manifest FILE] --n 50 --out DIR

    # tier 1, GPU: always under the shared card lock
    CUDA_VISIBLE_DEVICES=0 flock LOCKFILE pr0_stage_split.py run --root DIR ... \\
        --det-plan PLAN --pe-plan PLAN --gpu-index 0 --reps 3 --out DIR
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cpu_reference as reference
import harness_common as common
import numpy as np


sys.path.insert(0, str(common.REPO_ROOT))


def numpy_resample(
    frame: np.ndarray, size: tuple[int, int], *, antialias: bool = True
) -> np.ndarray:
    """Separable triangle-filter resample with the shipped weights (CPU, dense matrices)."""
    common.use_shipped_prep_modules()
    from prep_geometry import filter_weights

    h, w = frame.shape[:2]
    out = frame.astype(np.float32)
    for axis, (n_in, n_out) in enumerate(((h, size[1]), (w, size[0]))):
        first, count, weights = filter_weights(n_in, n_out, antialias=antialias)
        dense = np.zeros((n_out, n_in), np.float32)
        for i in range(n_out):
            dense[i, first[i] : first[i] + count[i]] = weights[i, : count[i]]
        out = (
            np.tensordot(dense, out, axes=([1], [axis]))
            if axis == 0
            else np.tensordot(out, dense, axes=([1], [1])).transpose(0, 2, 1)
        )
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def cpu_letterbox(frame: np.ndarray, net: int = 640, fill: int = 114) -> np.ndarray:
    common.use_shipped_prep_modules()
    from prep_geometry import letterbox_geometry

    lb = letterbox_geometry(frame.shape[1], frame.shape[0], net)
    canvas = np.full((net, net, 3), fill, np.uint8)
    canvas[lb.top : lb.top + lb.new_h, lb.left : lb.left + lb.new_w] = numpy_resample(
        frame, (lb.new_w, lb.new_h)
    )
    return np.ascontiguousarray(canvas.transpose(2, 0, 1))


def cpu_smoke(files: list[Path]) -> dict[str, Any]:
    """Tier 0: decode, references, the shipped weights on the CPU, the tensor gate, determinism."""
    rows = []
    for path in files:
        frame = reference.pillow_frame(path)
        ref = reference.letterbox_ref(frame)
        a, b = cpu_letterbox(frame), cpu_letterbox(frame)
        gate = reference.tensor_gate(ref, a)
        rows.append(
            {
                'mp': frame.shape[0] * frame.shape[1] / 1e6,
                'mean_abs': gate.mean_abs,
                'max_abs': gate.max_abs,
                'gate': gate.passed,
                'deterministic': bool(np.array_equal(a, b)),
            }
        )
    return {
        'tier': 0,
        'items': len(rows),
        'gate_pass': sum(r['gate'] for r in rows),
        'deterministic': all(r['deterministic'] for r in rows),
        'mean_abs_max': max(r['mean_abs'] for r in rows),
        'max_abs_max': max(r['max_abs'] for r in rows),
        'rows': rows,
    }


def parity_rows(results: list[Any], frames: list[np.ndarray]) -> list[dict[str, Any]]:
    out = []
    for res, frame in zip(results, frames, strict=True):
        lb = reference.tensor_gate(reference.letterbox_ref(frame), res.canvas[0])
        sq = reference.tensor_gate(reference.whole_square_ref(frame), res.squares[0])
        crops = [
            reference.tensor_gate(reference.crop_square_ref(frame, r), res.squares[k + 1])
            for k, r in enumerate(res.rects)
        ]
        out.append(
            {
                'letterbox': [lb.mean_abs, lb.max_abs, lb.passed],
                'whole': [sq.mean_abs, sq.max_abs, sq.passed],
                'crops': [[c.mean_abs, c.max_abs, c.passed] for c in crops],
            }
        )
    return out


def run_gpu(args: argparse.Namespace, files: list[Path]) -> dict[str, Any]:
    from shipped_pipeline import STAGES, ShippedPipeline
    from trt_runner import TrtModel

    pipe = ShippedPipeline(TrtModel(args.det_plan), TrtModel(args.pe_plan), workers=args.workers)
    warm = files[: args.warmup]
    count = files[args.warmup :]
    pipe.run_batch(warm)
    reps = []
    for rep in range(args.reps):
        common.drop_page_cache(count)
        start = time.perf_counter()
        with common.NvmlSampler(args.gpu_index) as nvml:
            results = pipe.run_batch(count)
        wall = max(r.finished for r in results) - start
        ok = [r for r in results if r.status == 'ok']
        reps.append(
            {
                'rep': rep,
                'load_1m': common.os.getloadavg()[0],
                'images': len(ok),
                'declined': len(results) - len(ok),
                'wall_s': wall,
                'images_per_s': len(ok) / wall,
                'nvml': nvml.summary(),
                'stage_ms_per_image': {
                    s: 1000 * sum(r.stages.get(s, 0.0) for r in ok) / max(1, len(ok))
                    for s in STAGES
                },
                'detections_per_image': float(np.mean([r.detections for r in ok])) if ok else 0.0,
            }
        )
    out: dict[str, Any] = {'tier': 1, 'reps': reps}
    if args.check:
        sample = count[: args.check]
        res1 = pipe.run_batch(sample, keep=True)
        res2 = pipe.run_batch(sample, keep=True)
        # files the router declines have no GPU output to check
        both = [
            (p, a, b)
            for p, a, b in zip(sample, res1, res2, strict=True)
            if a.status == 'ok' and b.status == 'ok'
        ]
        out['checked_images'] = len(both)
        out['parity'] = parity_rows(
            [a for _, a, _ in both], [reference.pillow_frame(p) for p, _, _ in both]
        )
        same = [(a, b) for _, a, b in both if a.rects == b.rects]
        # the detector's own run-to-run variation moves a crop rectangle by a pixel now and then;
        # the kernels are deterministic when the same rectangles give the same bytes
        out['deterministic'] = all(
            np.array_equal(a.canvas, b.canvas)
            and np.array_equal(a.squares, b.squares)
            and np.array_equal(a.sums, b.sums)
            for a, b in same
        )
        out['images_with_detector_rect_variation'] = len(both) - len(same)
    return out


def profile_gpu(args: argparse.Namespace, argv: list[str]) -> dict[str, Any]:
    """One traced repetition: GPU kernel ms per stage (NVTX ranges) and the kernel-busy share."""
    import nsys_stats

    inner = [
        sys.executable,
        str(Path(__file__).resolve()),
        'run',
        *[a for a in argv if a != 'profile'],
    ]
    inner[inner.index('--out') + 1] = str(
        args.out / 'traced'
    )  # keep the untraced run's file intact
    inner += ['--reps', '1']
    sqlite = nsys_stats.profile(inner, args.out / f'nsys_pr0_{args.label}')
    images = args.n
    ranges = nsys_stats.nvtx_kernel_ms(sqlite.with_suffix('.nsys-rep'))
    from shipped_pipeline import STAGES

    by_stage = {name: ranges.get(name, 0.0) / images for name in STAGES}
    # the NMS plug-in opens its own CUB range inside the detector call; it belongs to detect
    by_stage['detect'] += sum(v for k, v in ranges.items() if k.startswith('CCCL:')) / images
    total = sum(by_stage.values())
    return {
        'tier': 1,
        'traced_images': images,
        'gpu_kernel_ms_per_image_by_stage': by_stage,
        'gpu_kernel_ms_per_image_total': total,
        'pe_share_of_gpu_ms': by_stage['embed_pe'] / total if total else 0.0,
        'busy': nsys_stats.kernel_busy(sqlite),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('mode', choices=('cpu-smoke', 'run', 'profile'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--pin', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--n', type=int, default=50)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument(
        '--label', required=True, help='public set label for the result file, e.g. A, B, P'
    )
    p.add_argument('--det-plan', type=Path)
    p.add_argument('--pe-plan', type=Path)
    p.add_argument('--gpu-index', type=int)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--check', type=int, default=0, help='images checked against the CPU references')
    args = p.parse_args(argv)
    files = common.take(
        common.ordered_files(args.root, pin=args.pin, manifest=args.manifest), args.n
    )
    if args.mode == 'cpu-smoke':
        body = cpu_smoke(files)
    else:
        if None in (args.det_plan, args.pe_plan, args.gpu_index):
            p.error('run and profile need --det-plan, --pe-plan and --gpu-index')
        body = (
            run_gpu(args, files)
            if args.mode == 'run'
            else profile_gpu(args, argv if argv is not None else sys.argv[1:])
        )
    body['environment'] = common.environment(args.gpu_index if args.mode == 'run' else None)
    body['set'] = args.label
    common.write_json(args.out / f'pr0_{args.mode}_{args.label}.json', body)
    print(f'pr0 {args.mode} set {args.label}: wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
