#!/usr/bin/env python3
"""gpu:PR-2: resize and crop operations, control K-a (the shipped CuPy kernels).

    pr2_resize_ops.py cpu-smoke --root DIR [--pin F | --manifest F] --n 50 --label A --out DIR
    CUDA_VISIBLE_DEVICES=0 flock LOCK pr2_resize_ops.py run --root DIR ... --n 200 --label B \\
        --arms K-a K-b K-c --det-plan P --pe-plan P --gpu-index 0 --reps 3 --out DIR
    pr2_resize_ops.py profile ...   # the same run under nsys: GPU kernel ms per arm and operation
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


OPS = ('letterbox', 'whole_square', 'crop_squares', 'blur_sums')
BOX_TOL_PX = 1.0


def synthetic_rects(w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Tier-0 stand-in for detector boxes: a centre box and a left-third box."""
    return [(w // 4, h // 4, 3 * w // 4, 3 * h // 4), (0, h // 3, w // 3, 2 * h // 3)]


def cpu_smoke(files: list[Path]) -> dict[str, Any]:
    import pr0_stage_split

    rows = []
    for path in files:
        frame = reference.pillow_frame(path)
        rects = synthetic_rects(frame.shape[1], frame.shape[0])
        lb = reference.tensor_gate(
            reference.letterbox_ref(frame), pr0_stage_split.cpu_letterbox(frame)
        )
        a = reference.whole_square_ref(frame)
        crops = [reference.crop_square_ref(frame, r) for r in rects]
        rows.append(
            {
                'letterbox_gate': lb.passed,
                'square_shape': list(a.shape),
                'crop_shapes': [list(c.shape) for c in crops],
                'deterministic': bool(np.array_equal(a, reference.whole_square_ref(frame))),
            }
        )
    return {
        'tier': 0,
        'items': len(rows),
        'letterbox_gate_pass': sum(r['letterbox_gate'] for r in rows),
        'all_deterministic': all(r['deterministic'] for r in rows),
        'shapes_ok': all(
            r['square_shape'] == [3, 336, 336] and all(c == [3, 336, 336] for c in r['crop_shapes'])
            for r in rows
        ),
    }


def lap_variance(
    sums: Any, rects: list[tuple[int, int, int, int]], size: tuple[int, int]
) -> list[float]:
    """Laplacian variance per rectangle (frame first) from (sum, sum of squares) rows."""
    n = [size[0] * size[1]] + [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in rects]
    return [
        (c * q - s * s) / (c * c)
        for (s, q), c in zip(np.asarray(sums, np.float64).reshape(-1, 2), n, strict=True)
    ]


def downstream(
    detector: Any, encoder: Any, canvas: Any, squares: Any, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Boxes in frame pixels (from the arm's letterbox) and PE vectors (from the arm's squares)."""
    from prep_geometry import letterbox_geometry, undo_letterbox

    out = detector.run(canvas)
    n = int(out['num_dets'][0][0])
    lb = letterbox_geometry(size[0], size[1], 640)
    boxes = np.array(
        [undo_letterbox(tuple(b), 640, lb) for b in out['det_boxes'][0][:n].astype(np.float32)],
        np.float32,
    ).reshape(-1, 4)
    vec = encoder.run(squares)['image_embeddings']
    return boxes, vec / np.linalg.norm(vec, axis=1, keepdims=True)


def box_match(ctrl: np.ndarray, cand: np.ndarray) -> tuple[int, int, bool]:
    """(matched control boxes, control boxes, same count) with a 1 px tolerance in frame pixels."""
    hit = sum(
        bool(cand.size) and float(np.abs(cand - b).max(axis=1).min()) <= BOX_TOL_PX for b in ctrl
    )
    return hit, len(ctrl), len(ctrl) == len(cand)


def apply_op(arm: Any, op: str, frame: Any, rects: list[tuple[int, int, int, int]]) -> Any:
    if op == 'crop_squares':
        return arm.crop_squares(frame, rects)
    if op == 'blur_sums':
        return arm.blur_sums(frame, rects)
    if op == 'letterbox':
        return arm.letterbox(frame)
    return arm.whole_square(frame)


def stacked(o: dict[str, Any]) -> Any:
    import cupy as cp

    return cp.ascontiguousarray(cp.concatenate([o['whole_square'], o['crop_squares']]))


def score_arms(
    outs: dict[str, dict[str, Any]],
    ref: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    size: tuple[int, int],
    detector: Any,
    encoder: Any,
    gates: dict[str, Any],
    blur_rel: dict[str, list[float]],
    box_hits: dict[str, list[int]],
    pe_cos: dict[str, list[float]],
) -> None:
    """Tensor gates against the CPU references, blur sums and downstream boxes and PE vectors
    against the control arm K-a, for one image."""
    import cupy as cp

    ctrl = outs['K-a']
    ref_boxes, ref_vec = downstream(detector, encoder, ctrl['letterbox'], stacked(ctrl), size)
    ctrl_var = lap_variance(ctrl['blur_sums'], rects, size)
    for name, o in outs.items():
        gates[name]['letterbox'].append(
            list(reference.tensor_gate(reference.letterbox_ref(ref), cp.asnumpy(o['letterbox'][0])))
        )
        gates[name]['whole_square'].append(
            list(
                reference.tensor_gate(
                    reference.whole_square_ref(ref), cp.asnumpy(o['whole_square'][0])
                )
            )
        )
        gates[name]['crop_squares'].extend(
            list(
                reference.tensor_gate(
                    reference.crop_square_ref(ref, r), cp.asnumpy(o['crop_squares'][k])
                )
            )
            for k, r in enumerate(rects)
        )
        if 'blur_sums' in o:
            cand = lap_variance(o['blur_sums'], rects, size)
            blur_rel[name].extend(
                abs(a - b) / max(abs(a), 1e-12) for a, b in zip(ctrl_var, cand, strict=True)
            )
        boxes, vec = downstream(detector, encoder, o['letterbox'], stacked(o), size)
        hit, total, same = box_match(ref_boxes, boxes)
        counts = box_hits[name]
        counts[0] += hit
        counts[1] += total
        counts[2] += same
        counts[3] += 1
        pe_cos[name].extend(float(c) for c in np.sum(vec * ref_vec, axis=1)[: len(ref_vec)])


def run_gpu(args: argparse.Namespace, files: list[Path]) -> dict[str, Any]:
    import cupy as cp
    import resize_arms
    from decoder_arms import ShippedLanes
    from prep_geometry import crop_rect, letterbox_geometry, undo_letterbox
    from trt_runner import TrtModel

    detector, encoder = TrtModel(args.det_plan), TrtModel(args.pe_plan)
    decoder = ShippedLanes()
    arms = {n: resize_arms.build(n) for n in args.arms}
    ctrl = arms['K-a']
    times: dict[str, dict[str, list[float]]] = {n: {op: [] for op in OPS} for n in args.arms}
    gates: dict[str, dict[str, list[list[Any]]]] = {
        n: {op: [] for op in OPS[:3]} for n in args.arms
    }
    blur_rel: dict[str, list[float]] = {n: [] for n in args.arms}
    box_hits = {n: [0, 0, 0, 0] for n in args.arms}  # matched, total, items same count, items
    pe_cos: dict[str, list[float]] = {n: [] for n in args.arms}
    errors: dict[str, str] = {}
    for rep in range(args.reps):
        common.drop_page_cache(files)
        for path in files:
            frame = decoder.decode([path.read_bytes()])[0]
            h, w = frame.shape[:2]
            base = ctrl.letterbox(frame)
            lb = letterbox_geometry(w, h, 640)
            det = detector.run(base)
            rects = [
                crop_rect(undo_letterbox(tuple(b), 640, lb), w, h)
                for b in det['det_boxes'][0][: int(det['num_dets'][0][0])].astype(np.float32)
            ]
            ref = reference.pillow_frame(path) if rep == 0 else None
            outs: dict[str, dict[str, Any]] = {}
            names = list(args.arms) if rep % 2 == 0 else list(reversed(args.arms))
            for name in names:
                arm = arms[name]
                outs[name] = {}
                for op in OPS:
                    if name in errors and op in errors[name]:
                        continue
                    t0 = time.perf_counter()
                    try:
                        with resize_arms.traced(f'{name}/{op}'):
                            res = apply_op(arm, op, frame, rects)
                            cp.cuda.get_current_stream().synchronize()
                    except NotImplementedError as exc:
                        errors.setdefault(name, '')
                        errors[name] += f'{op}:{exc};'
                        continue
                    times[name][op].append(time.perf_counter() - t0)
                    outs[name][op] = res
            if rep == 0 and ref is not None:
                score_arms(
                    outs, ref, rects, (w, h), detector, encoder, gates, blur_rel, box_hits, pe_cos
                )
            del frame, outs
    return {
        'tier': 1,
        'images': len(files),
        'reps': args.reps,
        'errors': errors,
        'load_1m_end': common.os.getloadavg()[0],
        'wall_ms_per_op': {
            n: {op: 1000 * float(np.median(v)) if v else None for op, v in ops.items()}
            for n, ops in times.items()
        },
        'wall_ms_per_op_p90': {
            n: {op: 1000 * float(np.percentile(v, 90)) if v else None for op, v in ops.items()}
            for n, ops in times.items()
        },
        'tensor_gate': {
            n: {
                op: {
                    'n': len(g),
                    'passed': sum(x[2] for x in g),
                    'mean_abs_max': max((x[0] for x in g), default=None),
                    'max_abs_max': max((x[1] for x in g), default=None),
                }
                for op, g in ops.items()
            }
            for n, ops in gates.items()
        },
        'blur_rel_diff_max': {n: (max(v) if v else None) for n, v in blur_rel.items()},
        'p_box': {n: (b[0] / b[1] if b[1] else None) for n, b in box_hits.items()},
        'p_item': {n: (b[2] / b[3] if b[3] else None) for n, b in box_hits.items()},
        'pe_cosine': {
            n: (
                {'median': float(np.median(v)), 'p01': float(np.percentile(v, 1)), 'n': len(v)}
                if v
                else None
            )
            for n, v in pe_cos.items()
        },
    }


def profile_gpu(args: argparse.Namespace, argv: list[str]) -> dict[str, Any]:
    import nsys_stats

    inner = [
        sys.executable,
        str(Path(__file__).resolve()),
        'run',
        *[a for a in argv if a != 'profile'],
    ]
    inner[inner.index('--out') + 1] = str(args.out / 'traced')
    inner += ['--reps', '1']
    sqlite = nsys_stats.profile(inner, args.out / f'nsys_pr2_{args.label}')
    ranges = nsys_stats.nvtx_kernel_ms(sqlite.with_suffix('.nsys-rep'))
    return {
        'tier': 1,
        'images': args.n,
        'gpu_kernel_ms_per_image': {k: v / args.n for k, v in ranges.items() if '/' in k},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('mode', choices=('cpu-smoke', 'run', 'profile'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--pin', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--n', type=int, default=50)
    p.add_argument('--label', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arms', nargs='*', default=['K-a', 'K-b', 'K-c'])
    p.add_argument('--det-plan', type=Path)
    p.add_argument('--pe-plan', type=Path)
    p.add_argument('--gpu-index', type=int)
    p.add_argument('--reps', type=int, default=3)
    args = p.parse_args(argv)
    if 'K-a' not in args.arms:
        p.error('K-a is the control arm and must be listed')
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
    body['set'] = args.label
    body['environment'] = common.environment(args.gpu_index if args.mode != 'cpu-smoke' else None)
    common.write_json(args.out / f'pr2_{args.mode}_{args.label}.json', body)
    print(f'pr2 {args.mode} set {args.label}: wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
