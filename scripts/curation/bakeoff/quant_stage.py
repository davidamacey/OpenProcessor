"""Quantize + throughput stages of a bake-off job.

Moved out of ``bakeoff_runner.py``: exporting a trained checkpoint to
portable ONNX variants (``quantize.py``) that the same job then scores, and
the optional steady-state throughput sweep (``throughput.py``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


# The CoreML (Apple) export leg is not shipped: it needs a macOS host plus a
# host-side driver that is not part of this repository. A job asking for it
# gets this recorded as a failed stage instead of a silent skip or a crash.
COREML_UNAVAILABLE = (
    'CoreML export is not available in this build: it requires a macOS host and a '
    'CoreML export driver that this repository does not ship. Set quantize.coreml=false '
    '(score .mlpackage files you export yourself with --backend coreml).'
)


def _quant_root(quant: dict[str, Any], out_dir: Path) -> Path:
    """Where quantized artifacts go: ``quantize.out_root`` or ``<out_dir>/quant``."""
    return Path(quant['out_root']) if quant.get('out_root') else out_dir / 'quant'


def _quantize_and_variant_models(
    quant: dict[str, Any], datasets: list[dict[str, str]], out_dir: Path
) -> list[dict[str, Any]]:
    """Export a checkpoint to ONNX variants (``quantize.py``); return their model specs.

    Lets a single bake-off job export a trained model to portable ONNX
    (fp32/fp16/int8) and score those variants alongside the other models, so
    the matrix shows size / speed / accuracy with no manual step. The
    ``quantize`` block::

        {
            'model_id': 'my_model',
            'checkpoint': '/runs/.../best.pt',
            'formats': ['fp32_onnx', 'fp16_onnx', 'int8_onnx'],
            'n_calib': 1000,
            'calib_dataset': '/data/exports/<run>',  # default: first dataset
            'calib_split': 'train',
            'imgsz': 640,
            'out_root': '/data/quant',
        }  # default: <out_dir>/quant

    Raises :class:`quantize.QuantizeError` (or the exporter's own error) on
    failure; the caller records it in the job status.
    """
    from .quantize import run as quantize_run

    model_id = quant.get('model_id') or 'candidate'
    formats = quant.get('formats') or ['fp32_onnx', 'fp16_onnx', 'int8_onnx']
    out_root = _quant_root(quant, out_dir)
    calib = quant.get('calib_dataset') or (datasets[0]['path'] if datasets else None)
    checkpoint = quant.get('checkpoint')
    print(f'[bakeoff] quantize: exporting {model_id} {formats} from {checkpoint}', flush=True)
    quantize_run(
        model_id,
        list(formats),
        out_root,
        pt_override=Path(checkpoint) if checkpoint else None,
        calib_override=Path(calib) if calib else None,
        n_calib_override=quant.get('n_calib'),
        imgsz=int(quant.get('imgsz', 640)),
        calib_split=str(quant.get('calib_split', 'train')),
    )

    qdir = out_root / model_id
    # fp32/fp16 run on the GPU (portable, no engine build); INT8 QDQ runs on the
    # CPU EP (reliable; the GPU INT8 path needs the TensorRT EP) — its latency is
    # the edge/CPU datapoint while its mAP is EP-independent.
    variant_specs = [
        ('fp32_onnx', 'fp32.onnx', 'CUDAExecutionProvider,CPUExecutionProvider', 'FP32 ONNX'),
        (
            'fp16_onnx',
            'fp16.onnx',
            'CUDAExecutionProvider,CPUExecutionProvider',
            'FP16 ONNX (NVIDIA)',
        ),
        ('int8_onnx', 'int8_qdq.onnx', 'CPUExecutionProvider', 'INT8 QDQ ONNX'),
    ]
    out: list[dict[str, Any]] = []
    for fmt, fname, providers, label in variant_specs:
        if fmt not in formats:
            continue
        weights = qdir / fname
        if not weights.is_file():
            print(f'[bakeoff] quantize: missing {weights}, skipping', flush=True)
            continue
        out.append(
            {
                'backend': 'onnxruntime',
                'name': f'ours_{fmt.replace("_onnx", "")}_onnx',
                'weights': str(weights),
                'imgsz': quant.get('imgsz', 640),
                'device': 'cuda',
                'coords_normalized': False,
                'ort_providers': providers,
                'mode': 'full',
                'training_data': label,
            }
        )
    return out


def _run_throughput_sweep(
    quant: dict[str, Any], datasets: list[dict[str, str]], out_dir: Path
) -> None:
    """Steady-state img/s for every exported variant on CPU + GPU (best-effort).

    Runs ``throughput.py`` per variant/EP so the auto pipeline captures model
    *speed* (not just accuracy) -- the JSONs land in ``<out_dir>/throughput/``.
    """
    model_id = quant.get('model_id') or 'candidate'
    qdir = _quant_root(quant, out_dir) / model_id
    imgsz = str(quant.get('imgsz', 640))
    if not datasets:
        return
    images = Path(datasets[0]['path']) / 'images' / 'test'
    if not images.is_dir():
        print(f'[bakeoff] throughput: images dir {images} missing, skipping', flush=True)
        return
    tput_out = out_dir / 'throughput'
    # Each entry: backend, weights filename, label, ORT providers.
    runs = [
        ('ultralytics', 'source.pt', 'pt_gpu', None),
        ('onnxruntime', 'fp32.onnx', 'fp32_cuda', 'CUDAExecutionProvider,CPUExecutionProvider'),
        ('onnxruntime', 'fp32.onnx', 'fp32_cpu', 'CPUExecutionProvider'),
        ('onnxruntime', 'fp16.onnx', 'fp16_cuda', 'CUDAExecutionProvider,CPUExecutionProvider'),
        ('onnxruntime', 'fp16.onnx', 'fp16_cpu', 'CPUExecutionProvider'),
        ('onnxruntime', 'int8_qdq.onnx', 'int8_cuda', 'CUDAExecutionProvider,CPUExecutionProvider'),
        ('onnxruntime', 'int8_qdq.onnx', 'int8_cpu', 'CPUExecutionProvider'),
    ]
    for backend, fname, label, provs in runs:
        weights = qdir / fname
        if not weights.is_file():
            continue
        argv = [
            sys.executable,
            '-m',
            'scripts.curation.bakeoff.throughput',
            '--backend',
            backend,
            '--weights',
            str(weights),
            '--label',
            label,
            '--images',
            str(images),
            '--imgsz',
            imgsz,
            '--n-images',
            '200',
            '--warmup',
            '30',
            '--min-seconds',
            '6',
            '--out',
            str(tput_out),
        ]
        if backend == 'onnxruntime' and provs:
            argv += ['--ort-providers', provs]
        if backend == 'ultralytics':
            argv += ['--device', '0']
        try:
            subprocess.run(argv, check=True)
        except subprocess.CalledProcessError as exc:
            print(f'[bakeoff] throughput {label} failed: {exc}', flush=True)
