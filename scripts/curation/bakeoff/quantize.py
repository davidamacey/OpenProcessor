"""Export a trained YOLO checkpoint to portable, quantized ONNX artifacts.

From one trained ``.pt`` this produces the artifacts a bake-off job can score
side by side (the ``quantize`` block of a job spec, see ``bakeoff_runner``):

    fp32.onnx      baseline (no embedded NMS) -- the accuracy reference
    fp16.onnx      ~2x smaller; runs on ONNX Runtime CUDA EP with no engine build
    int8_qdq.onnx  portable QDQ INT8 (edge/CPU datapoint): ~4x smaller

FP16 ONNX comes from Ultralytics' own exporter (``half=True``). Ultralytics has
no INT8-ONNX path, so INT8 uses ONNX Runtime's static quantizer
(``quantize_static`` + QDQ), post-training-calibrated on images from a YOLO
dataset export (by default its ``train`` split -- never calibrate on the frozen
test split you score on). ``--formats plan`` optionally builds a TensorRT
engine with ``trtexec``; a ``.plan`` is locked to the GPU arch/TensorRT version
it was built on, so it is for a local Triton server only, never for
distribution.

Run inside the evaluator container (Ultralytics + onnx + onnxruntime-gpu)::

    python -m scripts.curation.bakeoff.quantize --model-id my_model \
        --pt /runs/<run>/weights/best.pt --calib-dataset /data/exports/<run> \
        --formats fp32_onnx,fp16_onnx,int8_onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .backends.yolo_post import letterbox, to_input_tensor


if TYPE_CHECKING:
    import numpy as np


logger = logging.getLogger('bakeoff.quantize')

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
ALL_FORMATS = ('fp32_onnx', 'fp16_onnx', 'int8_onnx', 'plan')
DEFAULT_IMGSZ = 640
DEFAULT_N_CALIB = 1000
DEFAULT_CALIB_SPLIT = 'train'


class QuantizeError(RuntimeError):
    """A quantize request that cannot run (bad input, missing weights/data).

    A ``RuntimeError`` rather than ``SystemExit`` so the bake-off watcher can
    catch it, record it in the job status and keep serving other jobs.
    """


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _artifact_entry(fmt: str, path: Path, **extra: Any) -> dict[str, Any]:
    return {
        'format': fmt,
        'path': path.name,
        'sha256': _sha256(path),
        'size_bytes': path.stat().st_size,
        'size_mb': round(path.stat().st_size / (1024 * 1024), 3),
        'nms': 'external',
        **extra,
    }


def select_calibration_images(
    dataset_root: Path,
    split: str,
    n: int,
    *,
    seed: int = 1234,
    background_frac: float = 0.15,
) -> list[Path]:
    """Pick ~``n`` calibration images from ``images/<split>``.

    Mostly frames with at least one labelled box, plus ``background_frac``
    empty frames so the quantizer also observes empty scenes. Deterministic
    via ``seed``.
    """
    img_dir = dataset_root / 'images' / split
    lbl_dir = dataset_root / 'labels' / split
    if not img_dir.is_dir():
        raise QuantizeError(f'calibration images dir not found: {img_dir}')
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise QuantizeError(f'no calibration images under {img_dir}')

    def _is_positive(img: Path) -> bool:
        lbl = lbl_dir / f'{img.stem}.txt'
        return lbl.is_file() and lbl.stat().st_size > 0

    positives = [p for p in images if _is_positive(p)]
    positive_set = set(positives)
    backgrounds = [p for p in images if p not in positive_set]

    rng = random.Random(seed)  # nosec B311 - reproducible sampling, not crypto
    rng.shuffle(positives)
    rng.shuffle(backgrounds)
    n_bg = min(len(backgrounds), round(n * background_frac))
    n_pos = min(len(positives), n - n_bg)
    chosen = positives[:n_pos] + backgrounds[:n_bg]
    if len(chosen) < n:  # top up if one bucket was short
        chosen_set = set(chosen)
        leftover = [p for p in images if p not in chosen_set]
        rng.shuffle(leftover)
        chosen += leftover[: n - len(chosen)]
    rng.shuffle(chosen)
    logger.info(
        'calibration set: %d images (%d positive, %d background) from %s',
        len(chosen),
        n_pos,
        n_bg,
        img_dir,
    )
    return chosen


def _make_reader(base_cls: type, images: list[Path], input_name: str, imgsz: int) -> Any:
    """CalibrationDataReader feeding letterboxed/255 NCHW float32 tensors.

    Preprocessing is the ONNX/Triton backends' own (``yolo_post``), so the
    calibration distribution matches real inference exactly.
    """
    import cv2

    class _YoloCalibrationReader(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            self._iter = iter(images)

        def get_next(self) -> dict[str, np.ndarray] | None:
            for path in self._iter:
                bgr = cv2.imread(str(path))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                lb, _scale, _pad = letterbox(rgb, imgsz)
                return {input_name: to_input_tensor(lb)}
            return None

        def rewind(self) -> None:
            self._iter = iter(images)

    return _YoloCalibrationReader()


def _ultralytics_export_onnx(pt_path: Path, out: Path, imgsz: int, *, half: bool) -> Path:
    """Export ``pt_path`` to ONNX (no NMS) via Ultralytics and move it to ``out``."""
    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else 'cpu'
    if half and device == 'cpu':
        raise QuantizeError('fp16 ONNX export requires a GPU (Ultralytics casts on device=0)')
    produced = YOLO(str(pt_path)).export(
        format='onnx',
        imgsz=imgsz,
        dynamic=False,
        simplify=True,
        half=half,
        opset=17,
        nms=False,
        device=device,
        verbose=False,
    )
    produced_path = Path(produced)
    if produced_path.resolve() != out.resolve():
        shutil.move(str(produced_path), str(out))
    logger.info('exported %s (%.2f MB)', out.name, out.stat().st_size / 1e6)
    return out


def quantize_int8_qdq(fp32_onnx: Path, out: Path, calib_images: list[Path], imgsz: int) -> Path:
    """Static QDQ INT8 post-training quantization of a YOLO ONNX graph.

    Only ``Conv``/``MatMul`` are quantized (per-channel weights, MinMax
    activations): an NMS-free decode tail (TopK / thresholds / box arithmetic)
    collapses to empty output when quantized, while the size/speed win lives
    in the backbone convolutions.
    """
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = out.with_name('._preprocessed.onnx')
    quant_pre_process(str(fp32_onnx), str(pre), skip_symbolic_shape=False)
    input_name = onnx.load(str(pre)).graph.input[0].name
    reader = _make_reader(CalibrationDataReader, calib_images, input_name, imgsz)
    logger.info('calibrating INT8 over %d images ...', len(calib_images))
    quantize_static(
        model_input=str(pre),
        model_output=str(out),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=['Conv', 'MatMul'],
        calibration_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
    pre.unlink(missing_ok=True)
    logger.info('wrote %s (%.2f MB)', out.name, out.stat().st_size / 1e6)
    return out


def export_trt_plan(onnx_path: Path, out: Path, *, int8: bool) -> Path | None:
    """Optional local-only TensorRT engine via ``trtexec`` (None if unavailable).

    TensorRT 11.1 is strongly typed: ``trtexec --fp16`` (and the
    ``BuilderFlag.FP16``/``platform_has_fast_fp16`` Python-API equivalents
    used elsewhere in ``export/trt_utils.py``) no longer exist. Precision
    is decided entirely by ``onnx_path``'s own tensor dtypes, so the
    caller is responsible for passing an already fp16-baked ONNX (e.g.
    ``fp16.onnx`` from Ultralytics' ``half=True`` export) when it wants a
    reduced-precision engine; this function never adds a precision flag.
    """
    trtexec = shutil.which('trtexec') or '/usr/src/tensorrt/bin/trtexec'
    if not Path(trtexec).exists():
        logger.warning('trtexec not found; skipping .plan')
        return None
    cmd = [trtexec, f'--onnx={onnx_path}', f'--saveEngine={out}']
    if int8:
        cmd.append('--int8')  # reads the QDQ scales embedded in int8_qdq.onnx
    logger.warning('building device-locked .plan (local Triton only): %s', ' '.join(cmd))
    subprocess.run(cmd, check=True)  # nosec B603 - fixed argv, no shell
    return out


def validate_request(
    formats: list[str], pt: Path | None, calib_root: Path | None
) -> tuple[list[str], Path]:
    """Check a quantize request before any heavy import; return (formats, pt)."""
    if not formats:
        raise QuantizeError('no formats requested')
    bad = [f for f in formats if f not in ALL_FORMATS]
    if bad:
        raise QuantizeError(f'unknown formats {bad}; valid: {list(ALL_FORMATS)}')
    if pt is None:
        raise QuantizeError('a checkpoint is required (quantize.checkpoint / --pt)')
    if not pt.is_file():
        raise QuantizeError(f'weights not found: {pt}')
    if 'int8_onnx' in formats and calib_root is None:
        raise QuantizeError('int8_onnx needs a calibration dataset (quantize.calib_dataset)')
    return formats, pt


def run(
    model_id: str,
    formats: list[str],
    out_root: Path,
    *,
    pt_override: Path | None,
    calib_override: Path | None,
    n_calib_override: int | None,
    imgsz: int = DEFAULT_IMGSZ,
    calib_split: str = DEFAULT_CALIB_SPLIT,
) -> dict[str, Any]:
    """Export ``formats`` for ``pt_override`` under ``out_root/model_id``; write a manifest."""
    if not model_id or '/' in model_id or model_id.startswith('.'):
        raise QuantizeError(f'invalid model_id: {model_id!r}')
    formats, pt_src = validate_request(formats, pt_override, calib_override)
    n_calib = n_calib_override or DEFAULT_N_CALIB

    out_dir = out_root / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Ultralytics writes the .onnx next to the weights, and /runs is mounted
    # read-only, so work from a copy in the output dir.
    pt_local = out_dir / 'source.pt'
    shutil.copy2(pt_src, pt_local)

    entries: list[dict[str, Any]] = []
    fp32_path = out_dir / 'fp32.onnx'
    if {'fp32_onnx', 'int8_onnx', 'plan'} & set(formats):
        _ultralytics_export_onnx(pt_local, fp32_path, imgsz, half=False)
        if 'fp32_onnx' in formats:
            entries.append(_artifact_entry('fp32_onnx', fp32_path))

    if 'fp16_onnx' in formats:
        fp16_path = _ultralytics_export_onnx(pt_local, out_dir / 'fp16.onnx', imgsz, half=True)
        entries.append(_artifact_entry('fp16_onnx', fp16_path, input_dtype='float16'))

    calib_images: list[Path] = []
    if 'int8_onnx' in formats:
        assert calib_override is not None  # validate_request guarantees it
        calib_images = select_calibration_images(calib_override, calib_split, n_calib)
        int8_path = quantize_int8_qdq(fp32_path, out_dir / 'int8_qdq.onnx', calib_images, imgsz)
        entries.append(
            _artifact_entry(
                'int8_onnx',
                int8_path,
                calib_method='minmax',
                per_channel=True,
                n_calib=len(calib_images),
            )
        )
        (out_dir / 'calibration_manifest.json').write_text(
            json.dumps([str(p) for p in calib_images], indent=2), encoding='utf-8'
        )

    if 'plan' in formats:
        # TRT 11.1 builds follow the source ONNX's own dtypes (no trtexec
        # --fp16 flag anymore), so prefer the fp16-baked ONNX when one was
        # produced -- otherwise the plan silently stays FP32/TF32.
        int8_src = out_dir / 'int8_qdq.onnx'
        fp16_src = out_dir / 'fp16.onnx'
        if int8_src.is_file():
            src = int8_src
        elif fp16_src.is_file():
            src = fp16_src
        else:
            src = fp32_path
        plan = export_trt_plan(src, out_dir / 'model.plan', int8=src == int8_src)
        if plan is not None:
            entries.append(_artifact_entry('plan', plan, distributable=False))

    manifest = {
        'model_id': model_id,
        'source_pt': str(pt_src),
        'source_pt_sha256': _sha256(pt_local),
        'imgsz': imgsz,
        'created_at': datetime.now(UTC).isoformat(),
        'calibration': {
            'dataset': str(calib_override),
            'split': calib_split,
            'n_images': len(calib_images),
            'selection': 'positive-weighted-with-backgrounds',
        }
        if calib_images
        else None,
        'artifacts': entries,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    logger.info('wrote manifest with %d artifact(s) -> %s', len(entries), out_dir)
    return manifest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--model-id', required=True, help='Output subdirectory / artifact name')
    p.add_argument('--pt', type=Path, required=True, help='Trained .pt weights')
    p.add_argument(
        '--formats',
        default='fp32_onnx,fp16_onnx,int8_onnx',
        help=f'Comma-separated subset of {ALL_FORMATS}',
    )
    p.add_argument('--out-root', type=Path, default=Path('./data/quant'))
    p.add_argument('--calib-dataset', type=Path, help='YOLO export root for INT8 calibration')
    p.add_argument('--calib-split', default=DEFAULT_CALIB_SPLIT)
    p.add_argument('--n-calib', type=int, default=DEFAULT_N_CALIB)
    p.add_argument('--imgsz', type=int, default=DEFAULT_IMGSZ)
    args = p.parse_args(argv)
    try:
        run(
            args.model_id,
            [f.strip() for f in args.formats.split(',') if f.strip()],
            args.out_root,
            pt_override=args.pt,
            calib_override=args.calib_dataset,
            n_calib_override=args.n_calib,
            imgsz=args.imgsz,
            calib_split=args.calib_split,
        )
    except QuantizeError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
