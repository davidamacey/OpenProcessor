#!/usr/bin/env python3
"""
Export SCRFD face detection model to TensorRT for Triton deployment.

SCRFD (Sample and Computation Redistribution for Face Detection) from InsightFace
provides face detection with 5-point landmarks, enabling proper Umeyama alignment
for ArcFace recognition.

Model: SCRFD-10G with Batch Normalization and Keypoints (scrfd_10g_bnkps)
- Input: [B, 3, 640, 640] FP32, RGB, normalized (x-127.5)/128
- Output: 9 tensors across 3 FPN strides (8, 16, 32):
  - score_{8,16,32}: face confidence per anchor
  - bbox_{8,16,32}: distance predictions (left, top, right, bottom)
  - kps_{8,16,32}: landmark offsets (5 points x 2 coords)
- WiderFace: 95.2% Easy / 93.9% Medium / 83.1% Hard

Key steps:
1. Download ONNX model from HuggingFace
2. Fix batch dimension for dynamic batching
3. Validate with ONNX Runtime
4. Convert to TensorRT FP16
5. Create Triton configuration

Usage:
    # From api container:
    docker compose exec api python /app/export/export_scrfd.py

    # From host with venv:
    python export/export_scrfd.py

    # Skip download if ONNX exists:
    python export/export_scrfd.py --skip-download

    # ONNX-only (no TensorRT):
    python export/export_scrfd.py --skip-trt
"""

import argparse
import logging
import sys

from scrfd_constants import MAX_BATCH_SIZE, ONNX_DIR, ONNX_FILENAME, OUTPUT_DIR, TRITON_MODEL_NAME
from scrfd_onnx import (
    analyze_onnx,
    benchmark_onnx,
    check_onnx_inference,
    download_onnx,
    make_batch_dynamic,
)
from scrfd_triton import convert_to_tensorrt, create_triton_config


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Export SCRFD face detection to TensorRT for Triton',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (download + export):
  python export/export_scrfd.py

  # Skip download (ONNX already exists):
  python export/export_scrfd.py --skip-download

  # ONNX validation only (no TensorRT):
  python export/export_scrfd.py --skip-trt

  # With benchmark:
  python export/export_scrfd.py --benchmark
""",
    )
    parser.add_argument('--skip-download', action='store_true', help='Skip ONNX download')
    parser.add_argument('--skip-trt', action='store_true', help='Skip TensorRT conversion')
    parser.add_argument('--force-download', action='store_true', help='Re-download ONNX')
    parser.add_argument('--benchmark', action='store_true', help='Run ONNX benchmark')
    parser.add_argument('--no-fp16', action='store_true', help='Disable FP16')
    parser.add_argument('--max-batch', type=int, default=MAX_BATCH_SIZE, help='Max batch size')
    args = parser.parse_args()

    logger.info('=' * 60)
    logger.info('SCRFD Face Detection Export')
    logger.info('=' * 60)

    onnx_path = ONNX_DIR / ONNX_FILENAME
    plan_dir = OUTPUT_DIR / TRITON_MODEL_NAME / '1'
    plan_path = plan_dir / 'model.plan'

    # Step 1: Download ONNX
    if not args.skip_download and not download_onnx(onnx_path, force=args.force_download):
        return 1

    if not onnx_path.exists():
        logger.error(f'ONNX not found: {onnx_path}')
        logger.info('Run: python export/download_face_models.py --models scrfd_10g_bnkps')
        return 1

    # Step 2: Analyze
    analyze_onnx(onnx_path)

    # Step 3: Make batch dynamic
    dynamic_onnx = make_batch_dynamic(onnx_path)

    # Step 4: Test inference
    check_onnx_inference(dynamic_onnx, batch_size=1)

    # Step 5: Benchmark
    if args.benchmark:
        benchmark_onnx(dynamic_onnx)

    # Step 6: Convert to TensorRT
    if not args.skip_trt:
        success = convert_to_tensorrt(
            dynamic_onnx,
            plan_path,
            fp16=not args.no_fp16,
            max_batch_size=args.max_batch,
        )
        if not success:
            return 1

        # Step 7: Create Triton config
        create_triton_config(plan_path, max_batch_size=args.max_batch)

    logger.info('')
    logger.info('=' * 60)
    logger.info('SCRFD Export Complete!')
    logger.info('=' * 60)
    logger.info(f'  ONNX: {onnx_path}')
    if not args.skip_trt:
        logger.info(f'  TensorRT: {plan_path}')
        logger.info(f'  Triton config: {plan_path.parent.parent / "config.pbtxt"}')
    logger.info('')
    logger.info('Next steps:')
    logger.info('  1. Restart Triton: docker compose restart triton-server')
    logger.info('  2. Load model: make load-scrfd')
    logger.info('  3. Test: python tests/test_scrfd_pipeline.py')

    return 0


if __name__ == '__main__':
    sys.exit(main())
