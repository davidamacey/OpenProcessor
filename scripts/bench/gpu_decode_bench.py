#!/usr/bin/env python3
"""Decode throughput probes for the GPU decode plan (#217, ``docs/design/gpu_decode_plan.md``).

Subcommands, each writing one JSON file:

``cpu``      single-thread ms per image for the CPU decoders the product uses or could use
             (Pillow with and without DCT scaling, OpenCV full and reduced), and images/s of the
             product's decode pair (Pillow full + OpenCV 1/8) across N worker processes.
             Run it in the API image so the decoder builds are the product's.
``gpu``      images/s, host CPU-s per image, GPU busy share, memory and PCIe bytes of GPU
             decoders (nvImageCodec, torchvision/nvJPEG, DALI) and of a GPU preprocessing arm
             that also builds the detector letterbox, the PE whole-frame square and the blur
             sums. Run it in the Triton image (it ships DALI and nvImageCodec).
``pe-load``  drives the PE encoder in a Triton at saturation for a fixed time (uint8 batch 32)
             and reports inferences/s, to measure what decoding on the same card takes from PE.
``formats``  which decoder accepts which file: progressive, CMYK, greyscale, 12-bit, PNG, WebP,
             corrupt, huge, every EXIF orientation; output equality against Pillow.

Pure logic (headers, routing, stats) is in ``gpu_decode_lib.py``. GPU imports are local to the
functions that need them so ``cpu`` runs without a GPU.

    python scripts/bench/gpu_decode_bench.py cpu --images DIR --label B --limit 60 --out cpu_b.json
    python scripts/bench/gpu_decode_bench.py gpu --images DIR --label B --arms nvimgcodec:32 \\
        --seconds 30 --out gpu_b.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.bench import gpu_decode_lib as lib  # noqa: E402


def load_images(dirs: list[str], limit: int) -> list[tuple[str, bytes]]:
    out: list[tuple[str, bytes]] = []
    for d in dirs:
        names = sorted(p for p in Path(d).iterdir() if p.is_file())
        out.extend((str(p), p.read_bytes()) for p in names[: limit or None])
    return out


def image_meta(data: bytes) -> dict[str, Any]:
    h = lib.jpeg_header(data)
    return {
        'bytes': len(data),
        'width': h.width,
        'height': h.height,
        'mp': h.width * h.height / 1e6,
        'orientation': h.orientation,
        'progressive': h.progressive,
        'kind': h.kind,
    }


# ---------------------------------------------------------------------------------- CPU


def _pil(data: bytes, scale: int = 1) -> np.ndarray:
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(data))
    if scale > 1:
        img.draft('RGB', (img.width // scale, img.height // scale))
    img = ImageOps.exif_transpose(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.load()
    return np.asarray(img)


def _cv2(data: bytes, reduced: int = 1) -> np.ndarray:
    import cv2

    flag = {
        1: cv2.IMREAD_COLOR,
        2: cv2.IMREAD_REDUCED_COLOR_2,
        4: cv2.IMREAD_REDUCED_COLOR_4,
        8: cv2.IMREAD_REDUCED_COLOR_8,
    }[reduced]
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flag)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _product_pair(data: bytes) -> int:
    """What ingest decodes per image today: Pillow in full (+EXIF), OpenCV at 1/8 for PE."""
    return int(_pil(data).shape[0]) + int(_cv2(data, 8).shape[0])


CPU_DECODERS: dict[str, Callable[[bytes], np.ndarray]] = {
    'pil': _pil,
    'pil_draft2': lambda b: _pil(b, 2),
    'pil_draft4': lambda b: _pil(b, 4),
    'pil_draft8': lambda b: _pil(b, 8),
    'cv2': _cv2,
    'cv2_r2': lambda b: _cv2(b, 2),
    'cv2_r4': lambda b: _cv2(b, 4),
    'cv2_r8': lambda b: _cv2(b, 8),
}

_POOL_IMAGES: list[bytes] = []


def _pool_init(paths: list[str]) -> None:
    import cv2

    cv2.setNumThreads(1)
    _POOL_IMAGES.extend(Path(p).read_bytes() for p in paths)


def _pool_task(i: int) -> int:
    return _product_pair(_POOL_IMAGES[i % len(_POOL_IMAGES)])


def run_cpu(args: argparse.Namespace) -> dict[str, Any]:
    import cv2

    cv2.setNumThreads(1)
    images = load_images(args.images, args.limit)
    per_decoder: dict[str, list[float]] = {k: [] for k in CPU_DECODERS}
    rows = []
    for name, data in images:
        row: dict[str, Any] = {'name': Path(name).name, **image_meta(data)}
        for key, fn in CPU_DECODERS.items():
            fn(data)
            t = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                fn(data)
                t.append((time.perf_counter() - t0) * 1e3)
            row[key] = min(t)
            per_decoder[key].append(min(t))
        rows.append(row)
    paths = [name for name, _ in images]
    pools = []
    for procs in args.procs:
        n = max(procs * 4, len(paths))
        with ProcessPoolExecutor(procs, initializer=_pool_init, initargs=(paths,)) as ex:
            list(ex.map(_pool_task, range(procs * 2)))
            t0 = time.perf_counter()
            list(ex.map(_pool_task, range(n), chunksize=1))
            wall = time.perf_counter() - t0
        pools.append({'procs': procs, 'images': n, 'images_per_s': n / wall})
    return {
        'label': args.label,
        'n_images': len(images),
        'mp': lib.summarize([r['mp'] for r in rows]),
        'ms_per_image': {k: lib.summarize(v) for k, v in per_decoder.items()},
        'product_pair_pool': pools,
        'cv2_threads': cv2.getNumThreads(),
        'rows': rows,
    }


# ---------------------------------------------------------------------------------- GPU


class GpuSampler:
    """10 Hz samples of one card's utilisation and memory from ``nvidia-smi`` (the Triton image
    has no NVML binding). PCIe bytes come from a host-side ``nvidia-smi dmon -s t`` log."""

    def __init__(self, index: int = 0) -> None:
        self._cmd = [
            'nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used',
            '--format=csv,noheader,nounits',
            f'--id={index}',
            '-lms',
            '100',
        ]
        self.samples: list[tuple[float, int, int]] = []
        self._proc: subprocess.Popen[str] | None = None
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            parts = [x.strip() for x in line.split(',')]
            if len(parts) == 2 and parts[0].isdigit():
                self.samples.append((time.time(), int(parts[0]), int(parts[1])))

    def __enter__(self) -> GpuSampler:
        self._proc = subprocess.Popen(self._cmd, stdout=subprocess.PIPE, text=True)
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._proc is not None
        self._proc.terminate()
        self._t.join(timeout=5)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {}
        return {
            'util_pct': lib.summarize([float(s[1]) for s in self.samples]),
            'mem_peak_mib': max(s[2] for s in self.samples),
        }


def _sync() -> None:
    import torch

    torch.cuda.synchronize()


def _nvimgcodec_decoder(kind: str, threads: int, fancy: bool) -> Callable[[list[bytes]], Any]:
    from nvidia import nvimgcodec

    backends = {
        'hybrid': [nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU)],
        'gpu': [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY)],
        'any': None,
    }[kind]
    options = ':fancy_upsampling=1' if fancy else ':fancy_upsampling=0'
    dec = nvimgcodec.Decoder(
        device_id=0, max_num_cpu_threads=threads, backends=backends, options=options
    )
    params = nvimgcodec.DecodeParams(
        apply_exif_orientation=True, color_spec=nvimgcodec.ColorSpec.SRGB
    )

    def run(batch: list[bytes]) -> Any:
        out = dec.decode(batch, params=params)
        _sync()
        return out

    return run


def _torchvision_decoder() -> Callable[[list[bytes]], Any]:
    import torch
    import torchvision

    def run(batch: list[bytes]) -> Any:
        ts = [torch.frombuffer(bytearray(b), dtype=torch.uint8) for b in batch]
        out = torchvision.io.decode_jpeg(
            ts, device='cuda', mode=torchvision.io.ImageReadMode.RGB, apply_exif_orientation=True
        )
        _sync()
        return out

    return run


def _dali_decoder(batch: int, threads: int, preprocess: bool) -> Callable[[list[bytes]], Any]:
    from nvidia.dali import fn, pipeline_def, types

    @pipeline_def(batch_size=batch, num_threads=threads, device_id=0, prefetch_queue_depth=1)
    def pipe() -> Any:
        jpegs = fn.external_source(name='jpegs', dtype=types.UINT8, ndim=1)
        img = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB, hw_decoder_load=0.0)
        if not preprocess:
            return img
        det = fn.resize(img, resize_longer=640, antialias=True, interp_type=types.INTERP_LINEAR)
        det = fn.pad(det, fill_value=114, shape=(640, 640, 3), align=None)
        pe = fn.resize(img, resize_shorter=336, antialias=True, interp_type=types.INTERP_LINEAR)
        pe = fn.crop(pe, crop=(336, 336))
        return det, pe

    p = pipe()
    p.build()

    def run(items: list[bytes]) -> Any:
        arrays = [np.frombuffer(b, dtype=np.uint8) for b in items]
        while len(arrays) < batch:
            arrays.append(arrays[-1])
        p.feed_input('jpegs', arrays)
        out = p.run()
        _sync()
        return out

    return run


def gpu_preprocess(frames: list[Any]) -> list[tuple[Any, Any, tuple[int, int, int]]]:
    """Per decoded HWC uint8 CUDA frame, everything ingest derives from the full frame: the
    detector letterbox, the PE whole-frame square and the blur sums (``gpu_decode_kernels``)."""
    import torch

    from scripts.bench import gpu_decode_kernels as k

    out = []
    for f in frames:
        t = torch.as_tensor(f, device='cuda')
        canvas, _, _ = k.letterbox_u8(t)
        out.append((canvas, k.whole_frame_u8(t), k.lap_sums(t)))
    _sync()
    return out


def _make_arm(arm: str, threads: int, batch: int) -> Callable[[list[bytes]], Any]:
    if arm in ('nvimgcodec', 'nvimgcodec_gpu', 'nvimgcodec_fancy', 'nvimgcodec_full'):
        kind = 'gpu' if arm == 'nvimgcodec_gpu' else 'any'
        dec = _nvimgcodec_decoder(kind, threads, fancy=arm == 'nvimgcodec_fancy')
        if arm != 'nvimgcodec_full':
            return dec
        return lambda b: gpu_preprocess(dec(b))
    if arm == 'nvimgcodec_hybrid':
        return _nvimgcodec_decoder('hybrid', threads, fancy=False)
    if arm == 'torchvision':
        return _torchvision_decoder()
    if arm in ('dali', 'dali_full'):
        return _dali_decoder(batch, threads, preprocess=arm == 'dali_full')
    msg = f'unknown arm {arm}'
    raise ValueError(msg)


def run_gpu(args: argparse.Namespace) -> dict[str, Any]:
    images = load_images(args.images, args.limit)
    data = [b for _, b in images]
    raw = sum(image_meta(b)['width'] * image_meta(b)['height'] * 3 for b in data)
    results = []
    for spec in args.arms:
        arm, _, rest = spec.partition(':')
        batch = int(rest or 1)
        fn = _make_arm(arm, args.threads, batch)
        batches = [data[i : i + batch] for i in range(0, len(data), batch)]
        for b in batches[: max(2, len(batches) // 10)]:
            fn(b)
        n_img = 0
        epoch0 = time.time()
        cpu0, t0 = time.process_time(), time.perf_counter()
        with GpuSampler(0) as sampler:
            while time.perf_counter() - t0 < args.seconds:
                for b in batches:
                    fn(b)
                    n_img += len(b)
                    if time.perf_counter() - t0 >= args.seconds:
                        break
        wall = time.perf_counter() - t0
        cpu = time.process_time() - cpu0
        results.append(
            {
                'arm': arm,
                'batch': batch,
                'threads': args.threads,
                'images': n_img,
                'wall_s': wall,
                'images_per_s': n_img / wall,
                'host_cpu_s_per_image': cpu / n_img,
                'host_cores': cpu / wall,
                'epoch_start': epoch0,
                'epoch_end': epoch0 + wall,
                **sampler.summary(),
            }
        )
        print(json.dumps(results[-1]), flush=True)
        del fn
    return {
        'label': args.label,
        'n_images': len(data),
        'jpeg_bytes_per_image': sum(map(len, data)) / len(data),
        'rgb_bytes_per_image': raw / len(data),
        'mp': lib.summarize([image_meta(b)['mp'] for b in data]),
        'arms': results,
    }


# ------------------------------------------------------------------------------ PE load


def run_pe_load(args: argparse.Namespace) -> dict[str, Any]:
    import tritonclient.grpc as grpcclient

    stop = time.perf_counter() + args.seconds
    counts = [0] * args.concurrency
    x = np.random.default_rng(0).integers(0, 256, size=(args.batch, 3, 336, 336), dtype=np.uint8)

    def worker(k: int) -> None:
        client = grpcclient.InferenceServerClient(args.url)
        inp = grpcclient.InferInput(args.input, list(x.shape), 'UINT8')
        inp.set_data_from_numpy(x)
        while time.perf_counter() < stop:
            client.infer(args.model, [inp])
            counts[k] += args.batch

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(args.concurrency)]
    series: list[tuple[float, int]] = []
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        series.append((time.time(), sum(counts)))
        time.sleep(0.5)
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    out = {'model': args.model, 'batch': args.batch, 'wall_s': wall, 'series': series}
    out['inferences_per_s'] = sum(counts) / wall
    print(json.dumps(out), flush=True)
    return out


# ------------------------------------------------------------------------------ formats


def make_formats(src: Path, out_dir: Path) -> list[str]:
    """Write the decoder-coverage set from one public JPEG: chroma subsampling, progressive,
    greyscale, CMYK, ICC, every EXIF orientation, odd and huge sizes, truncated and corrupted
    bitstreams, and non-JPEG formats. Returns the file names."""
    from PIL import Image, ImageCms

    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert('RGB')
    img.thumbnail((2048, 2048))
    jpeg: dict[str, dict[str, Any]] = {
        'base_420_q90': {'quality': 90, 'subsampling': 2},
        'chroma_422': {'quality': 90, 'subsampling': 1},
        'chroma_444': {'quality': 90, 'subsampling': 0},
        'progressive': {'quality': 90, 'progressive': True},
        'quality_50': {'quality': 50},
        'quality_100': {'quality': 100},
        'icc_srgb': {
            'quality': 90,
            'icc_profile': ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes(),
        },
    }
    names = []
    for name, kw in jpeg.items():
        img.save(out_dir / f'{name}.jpg', 'JPEG', **kw)
        names.append(f'{name}.jpg')
    img.convert('L').save(out_dir / 'gray.jpg', 'JPEG', quality=90)
    img.convert('CMYK').save(out_dir / 'cmyk.jpg', 'JPEG', quality=90)
    img.resize((1001, 667)).save(out_dir / 'odd_1001x667.jpg', 'JPEG', quality=90)
    img.resize((16, 16)).save(out_dir / 'tiny_16x16.jpg', 'JPEG', quality=90)
    img.resize((8000, 6000)).save(out_dir / 'huge_48mp.jpg', 'JPEG', quality=85)
    names += ['gray.jpg', 'cmyk.jpg', 'odd_1001x667.jpg', 'tiny_16x16.jpg', 'huge_48mp.jpg']
    for o in range(1, 9):
        exif = Image.Exif()
        exif[0x0112] = o
        img.save(out_dir / f'exif_orientation_{o}.jpg', 'JPEG', quality=90, exif=exif.tobytes())
        names.append(f'exif_orientation_{o}.jpg')
    base = (out_dir / 'base_420_q90.jpg').read_bytes()
    (out_dir / 'truncated_60pct.jpg').write_bytes(base[: int(len(base) * 0.6)])
    rng = np.random.default_rng(0)
    mid = bytearray(base)
    mid[len(mid) // 2 : len(mid) // 2 + 1024] = rng.integers(0, 256, 1024, dtype=np.uint8).tobytes()
    (out_dir / 'corrupt_middle.jpg').write_bytes(bytes(mid))
    names += ['truncated_60pct.jpg', 'corrupt_middle.jpg']
    for fmt, ext in (
        ('PNG', 'png'),
        ('WEBP', 'webp'),
        ('GIF', 'gif'),
        ('BMP', 'bmp'),
        ('TIFF', 'tiff'),
    ):
        img.save(out_dir / f'format.{ext}', fmt)
        names.append(f'format.{ext}')
    return names


def run_formats(args: argparse.Namespace) -> dict[str, Any]:
    """Each file through Pillow (the product), nvImageCodec and torchvision: accepted or the
    error, and the pixel difference to Pillow when both decode."""
    images = load_images(args.images, 0)
    nvi = _nvimgcodec_decoder('any', 4, fancy=False)
    nvi_fancy = _nvimgcodec_decoder('any', 4, fancy=True)
    tv = _torchvision_decoder()
    dali = _dali_decoder(1, 2, preprocess=False)
    decoders: dict[str, Callable[[bytes], np.ndarray]] = {
        'nvimgcodec': lambda b: np.asarray(nvi([b])[0].cpu()),
        'nvimgcodec_fancy': lambda b: np.asarray(nvi_fancy([b])[0].cpu()),
        'torchvision': lambda b: tv([b])[0].cpu().numpy().transpose(1, 2, 0),
        'dali': lambda b: np.asarray(dali([b])[0].as_cpu().at(0)),
    }
    rows = []
    for name, data in images:
        row: dict[str, Any] = {'name': Path(name).name, **image_meta(data)}
        try:
            ref = _pil(data)
            row['pil'] = list(ref.shape)
        except Exception as exc:
            ref, row['pil'] = None, f'error: {type(exc).__name__}: {exc}'[:160]
        for key, fn in decoders.items():
            try:
                arr = fn(data)
                row[key] = list(arr.shape)
                if ref is not None and arr.shape == ref.shape:
                    d = np.abs(arr.astype(np.int16) - ref.astype(np.int16))
                    row[key + '_mean_abs'] = float(d.mean())
                    row[key + '_max_abs'] = int(d.max())
            except Exception as exc:
                row[key] = f'error: {type(exc).__name__}: {exc}'[:160]
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {'rows': rows}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cpu')
    c.add_argument('--images', action='append', required=True)
    c.add_argument('--label', required=True)
    c.add_argument('--limit', type=int, default=0)
    c.add_argument('--reps', type=int, default=2)
    c.add_argument(
        '--procs', type=lambda s: [int(x) for x in s.split(',')], default=[1, 12, 24, 46]
    )
    g = sub.add_parser('gpu')
    g.add_argument('--images', action='append', required=True)
    g.add_argument('--label', required=True)
    g.add_argument('--limit', type=int, default=0)
    g.add_argument('--arms', type=lambda s: s.split(','), required=True, help='arm:batch,...')
    g.add_argument('--threads', type=int, default=8)
    g.add_argument('--seconds', type=float, default=20.0)
    pe = sub.add_parser('pe-load')
    pe.add_argument('--url', required=True)
    pe.add_argument('--model', default='pe_image_encoder')
    pe.add_argument('--input', default='images')
    pe.add_argument('--batch', type=int, default=32)
    pe.add_argument('--concurrency', type=int, default=6)
    pe.add_argument('--seconds', type=float, default=30.0)
    f = sub.add_parser('formats')
    f.add_argument('--images', action='append', required=True)
    mf = sub.add_parser('make-formats')
    mf.add_argument('--source', required=True, help='one public JPEG to derive the set from')
    for sp in (c, g, pe, f, mf):
        sp.add_argument('--out', required=True)
    args = p.parse_args(argv)
    if args.cmd == 'make-formats':
        print(make_formats(Path(args.source), Path(args.out)))
        return 0
    run = {'cpu': run_cpu, 'gpu': run_gpu, 'pe-load': run_pe_load, 'formats': run_formats}[args.cmd]
    result = run(args)
    result['host'] = {'cpu_count': os.cpu_count(), 'load_avg': os.getloadavg()}
    Path(args.out).write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
