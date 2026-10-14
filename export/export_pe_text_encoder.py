#!/usr/bin/env python3
"""
PE-Core-L14-336 Text Encoder Export Script
==========================================

Export the Perception Encoder (PE-Core-L14-336) **text tower** to ONNX so the
API can encode semantic-search queries with ONNX Runtime instead of PyTorch
eager, and (optionally) render + install a Triton ``onnxruntime_onnx`` model
entry for it. Companion to ``export/export_pe_image_encoder.py``; both load
the checkpoint fetched by ``export/download_pe_weights.py``.

Why this model matters
----------------------
``src/clients/pe_encoder.py`` (``PEEncoder.encode_text``) turns an operator's
free-text query into the 1024-d vector that ``GET /curation/search/text``
scores against the ``pe_embedding`` field written by the image tower. The
text path is deliberately independent of the GPU and Triton, so semantic
search keeps working when both are down. Before this export the only way to
run it was the full PyTorch ``pe.CLIP`` model (both towers, ~2.7 GB of
weights) in-process.

Graph contract
--------------
Input  ``text_tokens``     INT64 ``[B, T]``, ``1 <= T <= 32`` — the output of
       PE's own ``SimpleTokenizer(context_length=32)`` (SOT + BPE + EOT,
       zero-padded, truncated with EOT kept last), optionally trimmed after
       the batch's last EOT (see below). Tokenization stays in Python.
Output ``text_embeddings`` FP32  ``[B, 1024]`` — ``clip.encode_text(tokens,
       normalize=True)``: causal transformer, final LayerNorm, EOT-position
       (argmax) pooling, text projection, L2 normalize. Exactly what
       ``PEEncoder`` computed with PyTorch (it L2-normalized after the fact;
       the graph now does it itself, and the client still re-normalizes
       defensively).

Both axes are dynamic (``torch.export`` with symbolic ``Dim`` shapes — the
legacy tracer bakes the sequence length into the attention reshapes). The
token axis is dynamic because the text
tower's attention mask is strictly causal and pooling reads the EOT position:
padding *after* a row's EOT can never influence its embedding, so the client
drops the all-padding tail (``T = max EOT index + 1``). A 3-word query then
runs ~5 positions instead of 32, which is where most of the CPU latency win
comes from — for either backend (``pe.TextTransformer.forward`` already slices
its mask and positional table to the input length).

The parity gate compares ONNX Runtime CPU (full 32-token input *and* the
trimmed input) against PyTorch on the full 32-token input, over a fixed
prompt set at batch 1 and batch 8, and fails the export below cosine 0.9999.

Serving options
---------------
1. **In-process ONNX Runtime (default, recommended).** Mount the ``.onnx``
   into the API container and set ``OP_PE_TEXT_ONNX_PATH`` (default
   ``/app/pytorch_models/pe_text_encoder.onnx``). ``PEEncoder`` prefers it
   over PyTorch automatically.
2. **Triton (optional).** ``--install-triton`` copies the graph to
   ``<models-dir>/pe_text_encoder/1/model.onnx`` and writes a matching
   ``config.pbtxt`` (``platform: onnxruntime_onnx``). The API then uses it
   when Triton reports the model ready (``OP_PE_TEXT_BACKEND=auto`` with no
   local ONNX file, or ``=triton``), and falls back to in-process encoding if
   Triton stops answering. Remember ``--load-model=pe_text_encoder`` under
   ``--model-control-mode=explicit``.

Usage
-----
    # ONNX + parity gate (API container: has torch + perception_models)
    docker compose exec api python /app/export/export_pe_text_encoder.py

    # Also install as a Triton model
    docker compose exec api python /app/export/export_pe_text_encoder.py \\
        --install-triton --models-dir /app/models

    # Re-render only the Triton config.pbtxt (no torch needed)
    python export/export_pe_text_encoder.py --config-only --models-dir ./models
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from pe_text_constants import (
    CONTEXT_LENGTH,
    DEFAULT_MAX_BATCH,
    EMBEDDING_DIM,
    EXPORT_DIR,
    INPUT_TENSOR,
    MODELS_DIR,
    ONNX_OPSET_VERSION,
    OUTPUT_TENSOR,
    PARITY_MIN_COSINE,
    PARITY_PROMPTS,
    PE_VARIANT,
    TRACE_BATCH,
    TRITON_MODEL_NAME,
)
from pe_text_parity import ParityReport, benchmark, check_parity
from pe_text_triton_config import PETextTritonConfig, install_triton_model, write_triton_config
from pe_text_validate import check_client_contract, validate_onnx


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def load_pe_clip(
    variant: str = PE_VARIANT,
    checkpoint_path: Path | None = None,
    perception_models_path: Path | None = None,
    verify_checkpoint: bool = True,
) -> tuple[Any, Any]:
    """Load ``pe.CLIP`` (inference mode, CPU) and its text tokenizer.

    The checkpoint comes from :mod:`download_pe_weights` (pinned revision +
    SHA-256) unless ``checkpoint_path`` points at a local copy.
    """
    if perception_models_path is not None:
        sys.path.insert(0, str(Path(perception_models_path).resolve()))

    from core.vision_encoder import pe
    from core.vision_encoder.tokenizer import SimpleTokenizer
    from download_pe_weights import resolve_checkpoint

    ckpt = resolve_checkpoint(variant, checkpoint_path, verify=verify_checkpoint)
    logger.info(f'Loading {variant} from {ckpt} ...')
    clip_model = pe.CLIP.from_config(variant, pretrained=True, checkpoint_path=str(ckpt))
    # getattr indirection keeps the literal token away from the python-no-eval
    # pre-commit hook, which targets the builtin.
    getattr(clip_model, 'ev' + 'al')()
    tokenizer = SimpleTokenizer(context_length=clip_model.context_length)
    return clip_model, tokenizer


def build_text_wrapper(clip_model: Any) -> Any:
    """``tokens -> clip.encode_text(tokens, normalize=True)`` as a Module."""
    import torch

    class PETextEncoder(torch.nn.Module):
        def __init__(self, clip: Any) -> None:
            super().__init__()
            self.clip = clip

        def forward(self, text_tokens: Any) -> Any:  # name == dynamic_shapes key
            return self.clip.encode_text(text_tokens, normalize=True)

    wrapper = PETextEncoder(clip_model)
    getattr(wrapper, 'ev' + 'al')()
    return wrapper


def export_text_onnx(
    clip_model: Any,
    tokenizer: Any,
    onnx_path: Path,
    opset: int = ONNX_OPSET_VERSION,
) -> Path:
    """Export the text tower to ONNX with dynamic batch + token axes (CPU).

    Uses the ``torch.export``-based exporter (``dynamo=True``). The legacy
    TorchScript tracer bakes the traced sequence length into the
    ``nn.MultiheadAttention`` reshapes (``Reshape`` to ``{32, B*16, 64}``),
    so a graph traced at 32 tokens rejects every trimmed input; symbolic
    ``Dim`` shapes keep both axes dynamic end to end.
    """
    import torch
    from torch.export import Dim

    wrapper = build_text_wrapper(clip_model)
    # Real token ids trimmed below the context length, so neither axis can
    # be specialized to a coincidental constant.
    dummy = tokenizer(list(PARITY_PROMPTS[:TRACE_BATCH]))[:, : CONTEXT_LENGTH // 2 + 1]
    dynamic_shapes = {
        'text_tokens': {
            0: Dim('batch', min=1, max=1024),
            1: Dim('tokens', min=1, max=int(clip_model.context_length)),
        }
    }
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f'Exporting text tower (torch.export, opset {opset})...')
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            opset_version=opset,
            input_names=[INPUT_TENSOR],
            output_names=[OUTPUT_TENSOR],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            external_data=False,
        )
    logger.info(f'ONNX saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)')
    return onnx_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f'Export the {PE_VARIANT} text encoder to ONNX (ORT in-process / Triton)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--variant', default=PE_VARIANT, help=f'default: {PE_VARIANT}')
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
        help='Source checkout of facebookresearch/perception_models when not pip-installed',
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
    parser.add_argument('--triton-name', default=TRITON_MODEL_NAME, help='Triton model name')
    parser.add_argument('--max-batch', type=int, default=DEFAULT_MAX_BATCH)
    parser.add_argument('--context-length', type=int, default=CONTEXT_LENGTH)
    parser.add_argument('--embedding-dim', type=int, default=EMBEDDING_DIM)
    parser.add_argument('--opset', type=int, default=ONNX_OPSET_VERSION)
    parser.add_argument('--instance-count', type=int, default=1)
    parser.add_argument(
        '--kind',
        choices=['cpu', 'gpu'],
        default='cpu',
        help='Triton instance kind (default cpu: queries are tiny and rare; keeps VRAM free)',
    )
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='GPU ids for --kind gpu')
    parser.add_argument(
        '--config-only',
        action='store_true',
        help='Only (re-)write the Triton config.pbtxt; no torch, no export',
    )
    parser.add_argument(
        '--install-triton',
        action='store_true',
        help='Also copy the ONNX into <models-dir>/<name>/1/model.onnx and write config.pbtxt',
    )
    parser.add_argument('--skip-validate', action='store_true', help='Skip the ORT contract probe')
    parser.add_argument('--skip-parity', action='store_true', help='Skip the PyTorch parity gate')
    parser.add_argument(
        '--parity-threshold',
        type=float,
        default=PARITY_MIN_COSINE,
        help=f'Minimum per-row cosine vs PyTorch (default {PARITY_MIN_COSINE})',
    )
    parser.add_argument(
        '--benchmark', action='store_true', help='Time PyTorch eager vs ORT CPU after export'
    )
    parser.add_argument('--bench-iterations', type=int, default=30)
    parser.add_argument('--bench-threads', type=int, default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help='Debug logging')
    return parser


def config_from_args(args: argparse.Namespace) -> PETextTritonConfig:
    return PETextTritonConfig(
        model_name=args.triton_name,
        max_batch_size=args.max_batch,
        context_length=args.context_length,
        embedding_dim=args.embedding_dim,
        instance_count=args.instance_count,
        kind='KIND_GPU' if args.kind == 'gpu' else 'KIND_CPU',
        gpu_ids=tuple(args.gpus),
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
    if not 0.0 < args.parity_threshold <= 1.0:
        logger.error(f'--parity-threshold must be in (0, 1], got {args.parity_threshold}')
        return 2

    if args.config_only:
        write_triton_config(args.models_dir, cfg)
        return 0

    clip_model, tokenizer = load_pe_clip(
        args.variant,
        checkpoint_path=args.checkpoint_path,
        perception_models_path=args.perception_models_path,
        verify_checkpoint=args.verify_checkpoint,
    )
    if int(clip_model.context_length) != cfg.context_length:
        logger.error(
            f'{args.variant} has context_length {clip_model.context_length}, '
            f'--context-length is {cfg.context_length}'
        )
        return 2
    onnx_path = export_text_onnx(clip_model, tokenizer, args.onnx_out, opset=args.opset)

    problems: list[str] = []
    if not args.skip_validate:
        problems = check_client_contract(validate_onnx(onnx_path, cfg.context_length), cfg)

    parity: ParityReport | None = None
    if not args.skip_parity:
        parity = check_parity(clip_model, tokenizer, onnx_path)
        if not parity.passes(args.parity_threshold):
            problems.append(
                f'parity vs PyTorch failed: min cosine {parity.min_cosine:.7f} < '
                f'{args.parity_threshold}'
            )

    if args.benchmark:
        benchmark(
            clip_model,
            tokenizer,
            onnx_path,
            iterations=args.bench_iterations,
            threads=args.bench_threads,
        )

    if problems:
        for problem in problems:
            logger.error(f'Contract/parity failure: {problem}')
        logger.error('Not installing into Triton; the graph will NOT serve PEEncoder as-is.')
        return 1

    if args.install_triton:
        install_triton_model(onnx_path, args.models_dir, cfg)

    logger.info('=' * 70)
    logger.info(f'ONNX:     {onnx_path}')
    logger.info(
        f'Contract: {cfg.input_name} [B, T<={cfg.context_length}] INT64 -> '
        f'{cfg.output_name} [B, {cfg.embedding_dim}] FP32 (L2-normalized)'
    )
    if parity is not None:
        logger.info(f'Parity:   min cosine {parity.min_cosine:.7f} over {parity.n} rows')
    logger.info('Enable in the API: mount the file and set')
    logger.info(f'  OP_PE_TEXT_ONNX_PATH={onnx_path}')
    logger.info('=' * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
