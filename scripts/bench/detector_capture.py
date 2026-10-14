#!/usr/bin/env python3
"""Run end2end detector engines over a pinned image set and save their raw detections.

Input for ``detector_quality.py``. The preprocessing is the product's own
(``letterbox_to_square``, fill 114, 640 px), so the boxes are what ingest sees
before the confidence floor. Each model is ``NAME:uint8`` (engine scales in the
graph), ``NAME:fp32`` (this script divides by 255) or ``NAME:gpu_decode`` (the GPU ingest
GPU decode, ``gpu_image_prep``: it takes the encoded file, decodes and letterboxes on the GPU and
returns the detector rows with its letterbox). Needs ``tritonclient``,
``pillow`` and ``numpy`` (the API image has them); read-only against Triton.

    python scripts/bench/detector_capture.py --url 127.0.0.1:4961 \\
        --manifest scripts/datasets/manifests/coco_bench_2000.json \\
        --images /path/to/images --model det_old:fp32 --model det_u8:uint8 --out-dir dets/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.services.detection.geometry import letterbox_to_square, undo_letterbox  # noqa: E402  # import follows the sys.path setup above


OUTPUTS = ('num_dets', 'det_boxes', 'det_scores', 'det_classes')
NET = 640


def decode_row(
    num: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    scale: float,
    pad: tuple[float, float],
) -> list[list[float]]:
    out: list[list[float]] = []
    for k in range(int(num[0])):
        x1, y1, x2, y2 = undo_letterbox(tuple(float(v) * NET for v in boxes[k]), scale, pad)  # type: ignore[arg-type]
        out.append([x1, y1, x2, y2, float(scores[k]), int(classes[k])])
    return out


def gpu_decode_row(client: Any, model: str, data: bytes) -> list[list[float]]:
    """One image through the decoupled GPU decode. An image it declines (it would take the CPU
    path in ingest) stops the capture: the comparison needs every image from the GPU decode."""
    import queue

    import tritonclient.grpc as grpcclient

    answers: queue.Queue[Any] = queue.Queue()
    client.start_stream(callback=lambda result, error: answers.put((result, error)))
    try:
        inp = grpcclient.InferInput('jpeg', [len(data)], 'UINT8')
        inp.set_data_from_numpy(np.frombuffer(data, np.uint8))
        client.async_stream_infer(model, [inp])
        result, error = answers.get(timeout=120)
    finally:
        client.stop_stream()
    if error is not None:
        raise error
    status = result.as_numpy('status')[0]
    if status not in (b'ok', 'ok'):
        raise SystemExit(f'{model} declined an image: {status!r}')
    scale, pad_w, pad_h = (float(v) for v in result.as_numpy('letterbox'))
    boxes = result.as_numpy('det_boxes').reshape(-1, 4)
    return decode_row(
        np.array([len(boxes)]),
        boxes,
        result.as_numpy('det_scores'),
        result.as_numpy('det_classes'),
        scale,
        (pad_w, pad_h),
    )


def run(args: argparse.Namespace) -> int:
    import tritonclient.grpc as grpcclient

    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    rows = manifest['images']
    client = grpcclient.InferenceServerClient(args.url)
    specs: list[tuple[str, str]] = []
    for m in args.model:
        name, _, dtype = m.partition(':')
        if dtype not in ('uint8', 'fp32', 'gpu_decode'):
            raise SystemExit(f'--model {m!r}: dtype must be uint8, fp32 or gpu_decode')
        specs.append((name, dtype))
    results: dict[str, dict[str, Any]] = {n: {} for n, _ in specs}
    step = args.batch
    for start in range(0, len(rows), step):
        chunk = rows[start : start + step]
        lets = []
        for r in chunk:
            img = Image.open(Path(args.images) / r['file_name']).convert('RGB')
            lets.append(letterbox_to_square(img, NET))
        batch = np.concatenate([chw for chw, _s, _p in lets], axis=0)
        for name, dtype in specs:
            if dtype == 'gpu_decode':
                for r in chunk:
                    data = (Path(args.images) / r['file_name']).read_bytes()
                    results[name][str(r['image_id'])] = gpu_decode_row(client, name, data)
                continue
            data = batch if dtype == 'uint8' else batch.astype(np.float32) / np.float32(255.0)
            inp = grpcclient.InferInput('images', list(data.shape), dtype.upper())
            inp.set_data_from_numpy(data)
            res = client.infer(
                name, [inp], outputs=[grpcclient.InferRequestedOutput(o) for o in OUTPUTS]
            )
            o = {k: res.as_numpy(k) for k in OUTPUTS}
            for i, r in enumerate(chunk):
                _chw, scale, pad = lets[i]
                results[name][str(r['image_id'])] = decode_row(
                    o['num_dets'][i],
                    o['det_boxes'][i],
                    o['det_scores'][i],
                    o['det_classes'][i],
                    scale,
                    pad,
                )
        sys.stdout.write(f'{min(start + step, len(rows))}/{len(rows)}\n')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, _ in specs:
        (out_dir / f'{name}.json').write_text(json.dumps(results[name]), encoding='utf-8')
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--url', required=True, help='Triton gRPC host:port')
    p.add_argument('--manifest', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--model', action='append', required=True, metavar='NAME:uint8|fp32|gpu_decode')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--batch', type=int, default=16)
    return run(p.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
