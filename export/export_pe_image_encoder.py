#!/usr/bin/env python3
"""
PE-Core-L14-336 Image Encoder Export Script
===========================================

Export the Perception Encoder (PE-Core-L14-336) **vision tower** to ONNX and
render the Triton ``config.pbtxt`` the curation stack's PE client already
expects. The heavyweight TensorRT engine build lives in the two companion
shell scripts so it can run inside the Triton container (which is where
``trtexec`` lives):

1. ``export/export_pe_image_encoder.py`` — PyTorch -> ONNX (this file)
2. ``export/build_pe_trt.sh``            — ONNX -> TensorRT plan  (Path 1)
3. ``export/build_pe_ort_fallback.sh``   — ONNX served as-is      (Path 2)

Path 2 exists because PE's attention-pooling head has, on some TensorRT
releases, hit unsupported ops. Triton's ``onnxruntime_onnx`` backend serves
the same graph on the CUDA execution provider instead, at lower throughput
but with identical numerics.

Why this model matters
----------------------
``src/clients/pe_encoder.py`` (``PEEncoder.encode_images``) calls the Triton
model ``pe_image_encoder`` with a single UINT8 input named ``images`` shaped
``[B, 3, 336, 336]`` and reads a single FP32 output named
``image_embeddings`` shaped ``[B, 1024]``. That embedding is written to the
``pe_embedding`` field and is the vector behind curation semantic search,
near-duplicate detection, residual clustering and the embedding
visualization. Without this export a fresh deployment has no way to build
the model its own curation code requires.

The input tensor must be preprocessed exactly the way
``src/utils/pe_preprocess.py`` does it: resize shorter edge to
336, center-crop 336x336, CHW uint8 RGB pixels. The exported ONNX takes float
input already scaled to [0, 1] and PE-Core 0.5/0.5 normalized; the two build
scripts put that arithmetic in the graph (``export/uint8_input.py``) so the
served model takes the pixels. The exported graph L2-normalizes its own output;
``PEEncoder`` re-normalizes defensively, so either is safe.

Export methods
--------------
``--method perception-models`` (default)
    Loads the checkpoint (fetched + SHA-256 verified by
    ``export/download_pe_weights.py``, or ``--checkpoint-path``) through
    ``perception_models`` (already a declared
    project dependency; the pip distribution installs the top-level ``core``
    package) and runs ``torch.onnx.export`` on a thin wrapper whose I/O
    names are exactly ``images`` / ``image_embeddings``. This is the only
    method that produces the client's tensor contract without renaming.

``--method optimum``
    Exports the HuggingFace-native repackaging (``facebook/PE-Core-L14-336-hf``)
    through ``optimum.exporters.onnx``. Same weights, different packaging —
    but Optimum names its tensors ``pixel_values`` / ``image_embeds``, so
    the contract check below will report a mismatch and the rendered
    config.pbtxt has to be reconciled with the client (or the graph
    renamed) before it will serve. Kept because it needs neither a working
    ``perception_models`` install nor a GPU. The HF repo is gated: run
    ``huggingface-cli login`` or set ``HF_TOKEN`` first.

Usage
-----
    # ONNX + config.pbtxt (default: perception-models method)
    docker compose exec api python /app/export/export_pe_image_encoder.py

    # ONNX only, custom destination, no config.pbtxt
    python export/export_pe_image_encoder.py \\
        --onnx-out /tmp/pe_image_encoder.onnx --no-write-config

    # Re-render just the config.pbtxt for the ONNX-Runtime fallback path
    python export/export_pe_image_encoder.py --config-only \\
        --platform onnxruntime_onnx --models-dir ./models

Then build the engine:

    ONNX_PATH=/app/pytorch_models/pe_image_encoder.onnx \\
        bash export/build_pe_trt.sh        # Path 1 (preferred)
    ONNX_PATH=/app/pytorch_models/pe_image_encoder.onnx \\
        bash export/build_pe_ort_fallback.sh  # Path 2 (fallback)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pe_image_constants import (
    DEFAULT_MAX_BATCH,
    EMBEDDING_DIM,
    EXPORT_DIR,
    HF_MODEL_ID,
    IMAGE_SIZE,
    MODELS_DIR,
    ONNX_OPSET_VERSION,
    ORT_PLATFORM,
    PE_VARIANT,
    PLATFORMS,
    TRITON_MODEL_NAME,
    TRT_PLATFORM,
)
from pe_image_onnx_export import export_via_optimum, export_via_perception_models
from pe_image_triton_config import PETritonConfig, write_triton_config
from pe_image_validate import check_client_contract, validate_onnx


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (separated out so tests can exercise it)."""
    parser = argparse.ArgumentParser(
        description=f'Export the {PE_VARIANT} image encoder for Triton',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--method',
        choices=['perception-models', 'optimum'],
        default='perception-models',
        help='ONNX export backend (default: perception-models, the only one whose '
        'tensor names match the Triton client out of the box)',
    )
    parser.add_argument(
        '--variant',
        default=PE_VARIANT,
        help=f'perception_models checkpoint name (default: {PE_VARIANT})',
    )
    parser.add_argument(
        '--checkpoint-path',
        type=Path,
        default=None,
        help='Local PE checkpoint (.pt); default: pinned download via download_pe_weights.py',
    )
    parser.add_argument(
        '--no-verify-checkpoint',
        action='store_false',
        dest='verify_checkpoint',
        help='Skip the SHA-256 check of the checkpoint',
    )
    parser.add_argument(
        '--perception-models-path',
        type=Path,
        default=None,
        help='Path to a source checkout of facebookresearch/perception_models, '
        'used when the package is not pip-installed',
    )
    parser.add_argument(
        '--model',
        default=HF_MODEL_ID,
        help=f'HuggingFace model id for --method optimum (default: {HF_MODEL_ID})',
    )
    parser.add_argument(
        '--task',
        default='feature-extraction',
        help='Optimum task id. Try image-classification or '
        'zero-shot-image-classification if feature-extraction yields no pooled output.',
    )
    parser.add_argument(
        '--library',
        default=None,
        help='Optimum library override (e.g. timm); unset lets optimum infer',
    )
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        default=True,
        help='Allow optimum/transformers to execute the repo custom modeling code (default)',
    )
    parser.add_argument(
        '--no-trust-remote-code',
        action='store_false',
        dest='trust_remote_code',
        help='Refuse to execute custom modeling code',
    )
    parser.add_argument(
        '--onnx-out',
        type=Path,
        default=EXPORT_DIR / f'{TRITON_MODEL_NAME}.onnx',
        help=f'ONNX destination (default: {EXPORT_DIR / f"{TRITON_MODEL_NAME}.onnx"})',
    )
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=MODELS_DIR,
        help=f'Triton model repository root (default: {MODELS_DIR})',
    )
    parser.add_argument(
        '--triton-name',
        default=TRITON_MODEL_NAME,
        help=f'Triton model name (default: {TRITON_MODEL_NAME}; the client hardcodes this)',
    )
    parser.add_argument(
        '--platform',
        choices=list(PLATFORMS),
        default=TRT_PLATFORM,
        help=f'Triton platform to write into config.pbtxt (default: {TRT_PLATFORM}). '
        f'Use {ORT_PLATFORM} for the ONNX-Runtime fallback path.',
    )
    parser.add_argument(
        '--max-batch',
        type=int,
        default=DEFAULT_MAX_BATCH,
        help=f'Triton max_batch_size, must match the TRT profile (default: {DEFAULT_MAX_BATCH})',
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=IMAGE_SIZE,
        help=f'Input resolution (default {IMAGE_SIZE})',
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=EMBEDDING_DIM,
        help=f'Expected embedding width (default {EMBEDDING_DIM})',
    )
    parser.add_argument('--opset', type=int, default=ONNX_OPSET_VERSION, help='ONNX opset version')
    parser.add_argument(
        '--instance-count', type=int, default=1, help='Triton instance_group count (default 1)'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        default=None,
        help='pin the Triton instance group to these GPU ids (default: no pin, one instance per visible GPU)',
    )
    parser.add_argument(
        '--config-only',
        action='store_true',
        help='Only (re-)write config.pbtxt; skip the ONNX export entirely',
    )
    parser.add_argument(
        '--write-config',
        action='store_true',
        default=True,
        help='Write config.pbtxt into --models-dir (default)',
    )
    parser.add_argument(
        '--no-write-config',
        action='store_false',
        dest='write_config',
        help='Export the ONNX only; leave config.pbtxt alone',
    )
    parser.add_argument(
        '--skip-validate', action='store_true', help='Skip the ONNX Runtime probe after exporting'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Debug logging')
    return parser


def config_from_args(args: argparse.Namespace) -> PETritonConfig:
    """Build the (validated) Triton config from parsed CLI arguments."""
    return PETritonConfig(
        model_name=args.triton_name,
        platform=args.platform,
        max_batch_size=args.max_batch,
        image_size=args.image_size,
        embedding_dim=args.embedding_dim,
        instance_count=args.instance_count,
        gpu_ids=tuple(args.gpus) if args.gpus else None,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        cfg = config_from_args(args)
    except ValueError as exc:
        logger.error(f'Invalid arguments: {exc}')
        return 2
    if args.opset < 1:
        logger.error(f'--opset must be a positive int, got {args.opset}')
        return 2

    if args.config_only:
        write_triton_config(args.models_dir, cfg)
        return 0

    onnx_path = args.onnx_out
    if args.method == 'optimum':
        onnx_path = export_via_optimum(
            onnx_path,
            model_id=args.model,
            task=args.task,
            opset=args.opset,
            trust_remote_code=args.trust_remote_code,
            library_name=args.library,
        )
    else:
        onnx_path = export_via_perception_models(
            onnx_path,
            variant=args.variant,
            opset=args.opset,
            image_size=args.image_size,
            perception_models_path=args.perception_models_path,
            checkpoint_path=args.checkpoint_path,
            verify_checkpoint=args.verify_checkpoint,
        )
        # Same function, contiguous rotary halves: the engine is faster (export/pe_rope_layout.py).
        from pe_rope_layout import apply_if_supported

        apply_if_supported(onnx_path)

    problems: list[str] = []
    if not args.skip_validate:
        problems = check_client_contract(validate_onnx(onnx_path, args.image_size), cfg)

    if args.write_config:
        write_triton_config(args.models_dir, cfg)

    logger.info('=' * 70)
    logger.info(f'ONNX:     {onnx_path}')
    logger.info(f'Model:    {cfg.model_name} ({cfg.platform}, max_batch={cfg.max_batch_size})')
    logger.info(
        f'Contract: {cfg.input_name} [B, 3, {cfg.image_size}, {cfg.image_size}] UINT8 -> '
        f'{cfg.output_name} [B, {cfg.embedding_dim}] FP32'
    )
    logger.info('Next:')
    logger.info(f'  ONNX_PATH={onnx_path} bash export/build_pe_trt.sh          # Path 1: TensorRT')
    logger.info(f'  ONNX_PATH={onnx_path} bash export/build_pe_ort_fallback.sh # Path 2: ORT')
    logger.info('=' * 70)

    if problems:
        for problem in problems:
            logger.error(f'Contract mismatch: {problem}')
        logger.error('The exported graph will NOT serve PEEncoder as-is (see above).')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
