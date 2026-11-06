#!/usr/bin/env python3
"""gpu:PR-1: the decoder matrix. Control: the shipped nvJPEG GPU_HYBRID lanes.

    # tier 0 (no GPU): the harness, the router and the Pillow reference on 50 items
    pr1_decoders.py cpu-smoke --root DIR [--pin F | --manifest F] --n 50 --label A --out DIR

    # tier 1 (GPU, under the card lock): throughput + parity + coverage for the listed arms
    CUDA_VISIBLE_DEVICES=0 flock LOCK pr1_decoders.py bench --root DIR ... --n 200 --label B \\
        --arms control nvimgcodec:hybrid:8 dali:0:64 --gpu-index 0 --reps 3 --out DIR

    # coverage on the fixtures
    pr1_decoders.py coverage --root FIXTURE_DIR --arms ... --gpu-index 0 --out DIR

Counted work per repetition: blocks of 64 files, each block timed on its own call; arms alternate
in order inside every repetition; the page cache is dropped before each repetition.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cpu_reference as reference
import harness_common as common


BLOCK = 64
# what a decoder raises for a file it cannot decode: counted as a result, never a crash
DECODE_FAILURES = (RuntimeError, ValueError, OSError, IndexError, TypeError)


def read_all(files: list[Path]) -> list[bytes]:
    return [p.read_bytes() for p in files]


def block_rates(dec: Any, datas: list[bytes]) -> tuple[list[float], float]:
    """Images/s per full block of 64 and the process CPU seconds per image over the repetition."""
    rates: list[float] = []
    cpu0 = time.process_time()
    done = 0
    size = min(BLOCK, len(datas))  # a tier-0 sample is shorter than one registered block
    for s in range(0, len(datas) - size + 1, size):
        t0 = time.perf_counter()
        dec.decode(datas[s : s + size])
        rates.append(size / (time.perf_counter() - t0))
        done += size
    return rates, (time.process_time() - cpu0) / max(done, 1)


def parity(
    decs: dict[str, Any], files: list[Path], datas: list[bytes]
) -> dict[str, dict[str, Any]]:
    """Tensor gate against the Pillow decode (EXIF applied), per arm, for the files given.

    The reference is decoded once per file and compared on the GPU (one int16 difference per arm),
    the same gate as ``cpu_reference.tensor_gate``.
    """
    import cupy as cp

    acc: dict[str, dict[str, Any]] = {
        spec: {
            'gate_pass': 0,  # nosec B105
            'gate_fail': 0,
            'decode_errors': 0,
            'worst_mean_abs': 0.0,
            'worst_max_abs': 0,
        }
        for spec in decs
    }
    for path, data in zip(files, datas, strict=True):
        ref = cp.asarray(reference.pillow_frame(path)).astype(cp.int16)
        for spec, dec in decs.items():
            row = acc[spec]
            try:
                out = dec.decode([data])[0]
            except DECODE_FAILURES:
                out = None
            if out is None:
                row['decode_errors'] += 1
                continue
            if out.shape != ref.shape:
                row['gate_fail'] += 1
                continue
            diff = cp.abs(ref - out.astype(cp.int16))
            mean, peak = float(diff.mean()), int(diff.max())
            row['worst_mean_abs'] = max(row['worst_mean_abs'], mean)
            row['worst_max_abs'] = max(row['worst_max_abs'], peak)
            ok = mean < reference.GATE_MEAN_ABS and peak <= reference.GATE_MAX_ABS
            row['gate_pass' if ok else 'gate_fail'] += 1
        del ref
    return acc


def _simplejpeg(data: bytes) -> Any:
    import simplejpeg

    return simplejpeg.decode_jpeg(data, fastdct=False, fastupsample=False)


def _pillow_draft(data: bytes) -> Any:
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        img.draft('RGB', (640, 640))  # libjpeg scaled decode to at least the detector input
        return img.convert('RGB').load()


def cpu_columns(datas: list[bytes], threads: int = 16) -> dict[str, Any]:
    """The CPU reference columns (5c-K4): libjpeg-turbo through simplejpeg and Pillow ``draft``."""
    from concurrent.futures import ThreadPoolExecutor

    out: dict[str, Any] = {}
    with ThreadPoolExecutor(threads) as pool:
        for name, fn in (('simplejpeg', _simplejpeg), ('pillow_draft640', _pillow_draft)):
            list(pool.map(fn, datas[:BLOCK]))
            rates, cpu_s = block_rates(_Mapper(pool, fn), datas)
            out[name] = {
                'images_per_s_median': sorted(rates)[len(rates) // 2],
                'blocks': len(rates),
                'cpu_s_per_image': cpu_s,
            }
    return out


class _Mapper:
    def __init__(self, pool: Any, fn: Any) -> None:
        self.pool, self.fn = pool, fn

    def decode(self, datas: list[bytes]) -> list[Any]:
        return list(self.pool.map(self.fn, datas))


def cpu_smoke(files: list[Path]) -> dict[str, Any]:
    from decoder_arms import routed

    datas = read_all(files)
    verdicts = routed(datas)
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    mp = [reference.pillow_frame(p).shape for p in files[:10]]
    return {
        'tier': 0,
        'items': len(files),
        'router': counts,
        'pillow_decoded': len(mp),
        'cpu_columns': cpu_columns(datas, 8),
    }


def bench(args: argparse.Namespace, files: list[Path]) -> dict[str, Any]:
    from decoder_arms import build, routed

    datas = read_all(files)
    verdicts = routed(datas)
    passed = [d for d, v in zip(datas, verdicts, strict=True) if v == 'gpu']
    pass_files = [f for f, v in zip(files, verdicts, strict=True) if v == 'gpu']
    decs = {spec: build(spec) for spec in args.arms}
    for dec in decs.values():
        dec.decode(passed[:BLOCK])  # warm-up, not counted
    rates: dict[str, list[list[float]]] = {s: [] for s in args.arms}
    cpu: dict[str, list[float]] = {s: [] for s in args.arms}
    loads = []
    for rep in range(args.reps):
        order = list(args.arms) if rep % 2 == 0 else list(reversed(args.arms))
        for spec in order:
            common.drop_page_cache(pass_files)
            datas_rep = read_all(pass_files)  # reading is not the decoder's work
            loads.append(os.getloadavg()[0])
            r, c = block_rates(decs[spec], datas_rep)
            rates[spec].append(r)
            cpu[spec].append(c)
    out: dict[str, Any] = {
        'tier': 1,
        'decoded_images_per_arm': min(BLOCK, len(passed))
        + args.reps * (len(passed) // BLOCK * BLOCK),
        'images': len(files),
        'declined': len(files) - len(passed),
        'decline_reasons': {v: verdicts.count(v) for v in set(verdicts)},
        'load_1m_each_run': loads,
        'arms': {},
    }
    out['cpu_columns'] = cpu_columns(passed)
    ctrl = rates.get('control')
    checks = parity(decs, pass_files[: args.check], passed[: args.check]) if args.check else {}
    for spec in args.arms:
        flat = [x for rep in rates[spec] for x in rep]
        row: dict[str, Any] = {
            'images_per_s_median': float(sorted(flat)[len(flat) // 2]),
            'rep_medians': [sorted(r)[len(r) // 2] for r in rates[spec]],
            'spread': common.spread([sorted(r)[len(r) // 2] for r in rates[spec]]),
            'blocks': len(flat),
            'cpu_s_per_image': float(sum(cpu[spec]) / len(cpu[spec])),
        }
        if ctrl is not None and spec != 'control':
            row['ratio_vs_control'] = common.paired_bootstrap_ratio(
                flat, [x for rep in ctrl for x in rep]
            )
        row['parity'] = checks.get(spec, {})
        out['arms'][spec] = row
    return out


def profile(args: argparse.Namespace, argv: list[str]) -> dict[str, Any]:
    """GPU kernel and copy ms per decoded image of each arm: one traced repetition per arm."""
    import nsys_stats

    rows: dict[str, Any] = {}
    for spec in args.arms:
        inner = [sys.executable, str(Path(__file__).resolve()), 'bench']
        skip = False
        for tok in argv[1:]:
            if skip:
                skip = False
            elif tok in ('--arms', '--reps', '--check', '--out'):
                skip = True
            elif not tok.startswith('--') and tok in args.arms:
                continue
            else:
                inner.append(tok)
        traced = args.out / f'traced_{spec.replace(":", "_")}'
        inner += ['--arms', spec, '--reps', '1', '--check', '0', '--out', str(traced)]
        sqlite = nsys_stats.profile(
            inner, args.out / f'nsys_pr1_{args.label}_{spec.replace(":", "_")}'
        )
        busy = nsys_stats.kernel_busy(sqlite)
        decoded = json.loads((traced / f'pr1_bench_{args.label}.json').read_text())[
            'decoded_images_per_arm'
        ]
        rows[spec] = {
            'decoded_images': decoded,
            'gpu_kernel_ms_per_image': busy['kernel_ms'] / decoded,
            'gpu_memcpy_ms_per_image': busy['memcpy_ms'] / decoded,
            'busy_share': busy['busy_share'],
        }
    return {'tier': 1, 'arms': rows}


def coverage(args: argparse.Namespace, files: list[Path]) -> dict[str, Any]:
    """Per fixture and arm: decoded / declined / failed, and the oriented output vs Pillow."""
    import cupy as cp
    from decoder_arms import build, routed

    datas = read_all(files)
    verdicts = routed(datas)
    out: dict[str, Any] = {
        'router': dict(zip([f.name for f in files], verdicts, strict=True)),
        'arms': {},
    }
    for spec in args.arms:
        dec = build(spec)
        rows = {}
        for path, data in zip(files, datas, strict=True):
            try:
                frames = dec.decode([data])
                frame = frames[0]
            except DECODE_FAILURES as exc:
                rows[path.name] = f'error:{type(exc).__name__}'
                continue
            if frame is None:
                rows[path.name] = 'none'
                continue
            try:
                ref = reference.pillow_frame(path)
            except (
                OSError,
                ValueError,
                SyntaxError,
            ):  # Pillow cannot decode a deliberately broken file either
                rows[path.name] = 'decoded;no-reference'
                continue
            arr = cp.asnumpy(frame)
            gate = reference.tensor_gate(ref, arr) if arr.shape == ref.shape else None
            rows[path.name] = (
                'ok' if gate and gate.passed else f'mismatch:shape={arr.shape == ref.shape}'
            )
        out['arms'][spec] = rows
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('mode', choices=('cpu-smoke', 'bench', 'coverage', 'profile'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--pin', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--n', type=int, default=50)
    p.add_argument('--label', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arms', nargs='*', default=['control'])
    p.add_argument('--gpu-index', type=int)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--check', type=int, default=200, help='files compared with Pillow per arm')
    args = p.parse_args(argv)
    files = (
        common.take(common.ordered_files(args.root, pin=args.pin, manifest=args.manifest), args.n)
        if args.mode != 'coverage'
        else sorted(args.root.glob('*.jpg'))
    )
    body = {
        'cpu-smoke': lambda: cpu_smoke(files),
        'bench': lambda: bench(args, files),
        'coverage': lambda: coverage(args, files),
        'profile': lambda: profile(args, argv if argv is not None else sys.argv[1:]),
    }[args.mode]()
    body['set'] = args.label
    body['environment'] = common.environment(args.gpu_index)
    common.write_json(args.out / f'pr1_{args.mode}_{args.label}.json', body)
    print(f'pr1 {args.mode} set {args.label}: wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
