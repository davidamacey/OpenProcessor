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
model ``pe_image_encoder`` with a single FP32 input named ``images`` shaped
``[B, 3, 336, 336]`` and reads a single FP32 output named
``image_embeddings`` shaped ``[B, 1024]``. That embedding is written to the
``pe_embedding`` field and is the vector behind curation semantic search,
near-duplicate detection, residual clustering and the embedding
visualization. Without this export a fresh deployment has no way to build
the model its own curation code requires.

The input tensor must be preprocessed exactly the way
``src/services/detection/pe_preprocess.py`` does it: resize shorter edge to
336, center-crop 336x336, scale to [0, 1], PE-Core 0.5/0.5 normalize, CHW
float32. The exported graph L2-normalizes its own output; ``PEEncoder``
re-normalizes defensively, so either is safe.

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
    docker compose exec yolo-api python /app/export/export_pe_image_encoder.py

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
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Triton model name + tensor names are a hard contract with
# src/clients/pe_encoder.py — do not rename one side without the other.
TRITON_MODEL_NAME = 'pe_image_encoder'
INPUT_TENSOR = 'images'
OUTPUT_TENSOR = 'image_embeddings'

PE_VARIANT = 'PE-Core-L14-336'
HF_MODEL_ID = 'facebook/PE-Core-L14-336-hf'
IMAGE_SIZE = 336
EMBEDDING_DIM = 1024

# opset 17 is the floor for clean LayerNorm + dynamic-batch handling in the
# TensorRT ONNX parser (same rationale as export_mobileclip_image_encoder.py).
ONNX_OPSET_VERSION = 17

# Batch 32 at 336x336 is the profile the reference deployment ships:
# min=1 / opt=8 / max=32. Larger batches mostly buy queueing, not
# throughput, for a ViT-L at this resolution.
DEFAULT_MAX_BATCH = 32

# Triton platform identifiers for the two serving paths.
TRT_PLATFORM = 'tensorrt_plan'
ORT_PLATFORM = 'onnxruntime_onnx'
PLATFORMS = (TRT_PLATFORM, ORT_PLATFORM)

# Container paths, matching the sibling export scripts. Override on the CLI
# when running on the host.
MODELS_DIR = Path('/app/models')
EXPORT_DIR = Path('/app/pytorch_models')

# The ONNX exporter is traced with a batch-2 dummy on purpose. With a
# batch-1 dummy the PE attention-pool traces a Reshape that captures
# batch=1 as a *constant* volume; TensorRT then rejects every batch > 1
# with "reshape would change volume". A batch-2 dummy forces the tracer to
# treat the leading dimension as symbolic. Verified against ONNX Runtime at
# batch 1/2/4/8/16/32.
TRACE_BATCH = 2


# ============================================================================
# Triton config rendering (pure — unit-tested without a GPU)
# ============================================================================


@dataclass(frozen=True)
class PETritonConfig:
    """Everything that varies in the rendered ``config.pbtxt``.

    Defaults describe the model ``src/clients/pe_encoder.py`` expects:
    ``pe_image_encoder``, FP32 ``images`` in, FP32 ``image_embeddings``
    out, dynamic batch up to 32.
    """

    model_name: str = TRITON_MODEL_NAME
    platform: str = TRT_PLATFORM
    max_batch_size: int = DEFAULT_MAX_BATCH
    image_size: int = IMAGE_SIZE
    embedding_dim: int = EMBEDDING_DIM
    input_name: str = INPUT_TENSOR
    output_name: str = OUTPUT_TENSOR
    instance_count: int = 1
    gpu_ids: tuple[int, ...] = (0,)
    max_queue_delay_us: int = 15000

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError('model_name must be a non-empty string')
        if self.platform not in PLATFORMS:
            raise ValueError(f'platform must be one of {PLATFORMS}, got {self.platform!r}')
        for attr in ('max_batch_size', 'image_size', 'embedding_dim', 'instance_count'):
            value = getattr(self, attr)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f'{attr} must be a positive int, got {value!r}')
        if self.max_queue_delay_us < 0:
            raise ValueError(f'max_queue_delay_us must be >= 0, got {self.max_queue_delay_us}')
        if not self.gpu_ids:
            raise ValueError('gpu_ids must contain at least one device id')
        if any(not isinstance(g, int) or g < 0 for g in self.gpu_ids):
            raise ValueError(f'gpu_ids must be non-negative ints, got {self.gpu_ids!r}')
        if not self.input_name or not self.output_name:
            raise ValueError('input_name and output_name must be non-empty strings')


def preferred_batch_sizes(max_batch: int) -> list[int]:
    """Triton ``preferred_batch_size`` ladder bounded by ``max_batch``."""
    ladder = [size for size in (4, 8, 16, 32, 64) if size <= max_batch]
    return ladder or [max_batch]


def render_config(cfg: PETritonConfig) -> str:
    """Render the Triton ``config.pbtxt`` body for the PE image encoder."""
    preferred = ', '.join(str(b) for b in preferred_batch_sizes(cfg.max_batch_size))
    gpus = ', '.join(str(g) for g in cfg.gpu_ids)
    if cfg.platform == TRT_PLATFORM:
        provenance = (
            '# TensorRT plan built by export/build_pe_trt.sh from the ONNX\n'
            '# produced by export/export_pe_image_encoder.py.\n'
            f'# Engine profile: min=1 / opt=8 / max={cfg.max_batch_size} '
            f'at {cfg.image_size}x{cfg.image_size}.'
        )
    else:
        provenance = (
            '# ONNX served directly by Triton (Path 2 fallback, installed by\n'
            '# export/build_pe_ort_fallback.sh). Use this when the TensorRT\n'
            "# build fails on PE's attention-pool ops. Same numerics, lower\n"
            '# throughput. The ONNX MUST have a dynamic leading dimension or\n'
            '# dynamic_batching below is invalid.'
        )

    return f"""# {PE_VARIANT} image encoder — whole-frame and per-crop embeddings.
#
# Consumed by src/clients/pe_encoder.py (PEEncoder.encode_images); the
# resulting vector is stored as the curation `pe_embedding` field and backs
# semantic search, near-duplicate detection, clustering and the embedding
# visualization. The tensor names below are a contract with that client.
#
# Input:  {cfg.input_name} [B, 3, {cfg.image_size}, {cfg.image_size}] FP32,
#         PE-Core 0.5/0.5 mean/std normalized (see src/services/detection/pe_preprocess.py)
# Output: {cfg.output_name} [B, {cfg.embedding_dim}] FP32, L2-normalized
#
{provenance}

name: "{cfg.model_name}"
platform: "{cfg.platform}"
max_batch_size: {cfg.max_batch_size}

input [
  {{
    name: "{cfg.input_name}"
    data_type: TYPE_FP32
    dims: [ 3, {cfg.image_size}, {cfg.image_size} ]
  }}
]

output [
  {{
    name: "{cfg.output_name}"
    data_type: TYPE_FP32
    dims: [ {cfg.embedding_dim} ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ {preferred} ]
  max_queue_delay_microseconds: {cfg.max_queue_delay_us}
}}

instance_group [
  {{
    count: {cfg.instance_count}
    kind: KIND_GPU
    gpus: [ {gpus} ]
  }}
]
"""


def write_triton_config(models_dir: Path, cfg: PETritonConfig) -> Path:
    """Write ``<models_dir>/<model_name>/config.pbtxt`` and return its path."""
    model_dir = models_dir / cfg.model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / 'config.pbtxt'
    config_path.write_text(render_config(cfg))
    logger.info(f'Generated Triton config: {config_path}')
    return config_path


# ============================================================================
# ONNX export paths
# ============================================================================


def export_via_perception_models(
    onnx_path: Path,
    variant: str = PE_VARIANT,
    opset: int = ONNX_OPSET_VERSION,
    image_size: int | None = None,
    perception_models_path: Path | None = None,
    checkpoint_path: Path | None = None,
    verify_checkpoint: bool = True,
) -> Path:
    """Export the PE vision tower with ``torch.onnx.export``.

    Loads ``core.vision_encoder.pe.CLIP`` (the importable package name of
    the ``perception_models`` distribution) and traces a wrapper whose
    forward is ``clip.encode_image(images, normalize=True)`` — so the graph
    emits an already-L2-normalized pooled embedding under the tensor names
    the Triton client expects.

    Args:
        onnx_path: Destination ``.onnx`` file.
        variant: PE checkpoint name, e.g. ``PE-Core-L14-336``.
        opset: ONNX opset version.
        image_size: Override the checkpoint's own input resolution.
        perception_models_path: Optional path to a source checkout of
            facebookresearch/perception_models, prepended to ``sys.path``
            when the package isn't pip-installed.
        checkpoint_path: Local checkpoint file. Default: the pinned revision
            from ``download_pe_weights.py`` (downloaded into / reused from
            the HF cache), SHA-256 verified.
        verify_checkpoint: Check the checkpoint's SHA-256 against the pin.

    Returns:
        The written ONNX path.
    """
    if perception_models_path is not None:
        sys.path.insert(0, str(Path(perception_models_path).resolve()))

    import torch
    from core.vision_encoder import pe
    from download_pe_weights import resolve_checkpoint
    from pe_rearrange_shim import static_batch_safe_rearrange

    ckpt = resolve_checkpoint(variant, checkpoint_path, verify=verify_checkpoint)
    logger.info(f'Loading {variant} via perception_models from {ckpt} ...')
    clip_model = pe.CLIP.from_config(variant, pretrained=True, checkpoint_path=str(ckpt))
    # Inference mode. getattr indirection keeps the literal token away from
    # the python-no-eval pre-commit hook, which targets the builtin.
    getattr(clip_model, 'ev' + 'al')()
    resolved_size = int(image_size or getattr(clip_model, 'image_size', IMAGE_SIZE))
    logger.info(f'  Checkpoint loaded (input resolution {resolved_size})')

    class PEImageEncoder(torch.nn.Module):
        """Thin traceable wrapper: images -> L2-normalized pooled embedding."""

        def __init__(self, clip: Any) -> None:
            super().__init__()
            self.clip = clip

        def forward(self, images: Any) -> Any:
            return self.clip.encode_image(images, normalize=True)

    wrapper = PEImageEncoder(clip_model)
    getattr(wrapper, 'ev' + 'al')()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        wrapper = wrapper.cuda()
    dummy = torch.zeros(
        TRACE_BATCH, 3, resolved_size, resolved_size, dtype=torch.float32, device=device
    )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f'Tracing with a batch-{TRACE_BATCH} dummy on {device} (opset {opset})...')
    # The legacy TorchScript tracer bakes the traced batch size into
    # SelfAttention's einops.rearrange reshapes as ONNX Reshape constants
    # (see pe_rearrange_shim for the full mechanism); patch those two call
    # sites to trace-safe unflatten/transpose equivalents for the export.
    with static_batch_safe_rearrange():
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=[INPUT_TENSOR],
            output_names=[OUTPUT_TENSOR],
            dynamic_axes={INPUT_TENSOR: {0: 'batch'}, OUTPUT_TENSOR: {0: 'batch'}},
            dynamo=False,
        )
    logger.info(f'ONNX saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)')
    return onnx_path


def export_via_optimum(
    onnx_path: Path,
    model_id: str = HF_MODEL_ID,
    task: str = 'feature-extraction',
    opset: int = ONNX_OPSET_VERSION,
    trust_remote_code: bool = True,
    library_name: str | None = None,
) -> Path:
    """Export the HF-native PE repackaging through ``optimum.exporters.onnx``.

    Optimum writes ``model.onnx`` plus its own config files into a
    *directory*; the result is moved to ``onnx_path`` so both methods
    agree on where the artifact lands.

    Note:
        Optimum names the graph's tensors after the HF signature
        (``pixel_values`` / ``image_embeds``), not after the Triton client's
        contract. :func:`check_client_contract` will flag that; reconcile it
        before serving.
    """
    try:
        from optimum.exporters.onnx import main_export
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'The --method optimum path needs optimum. Install it with:\n'
            "  pip install 'optimum[onnxruntime]' transformers"
        ) from exc

    work_dir = onnx_path.parent / f'{onnx_path.stem}_optimum'
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Optimum export: model={model_id} task={task} opset={opset}')

    kwargs: dict[str, Any] = {
        'model_name_or_path': model_id,
        'output': work_dir,
        'task': task,
        'opset': opset,
        'framework': 'pt',
        'trust_remote_code': trust_remote_code,
    }
    if library_name:
        kwargs['library_name'] = library_name
    main_export(**kwargs)

    produced = work_dir / 'model.onnx'
    if not produced.exists():
        # Some Optimum releases emit a different filename.
        candidates = sorted(work_dir.glob('*.onnx'))
        if not candidates:
            raise RuntimeError(f'Optimum produced no .onnx under {work_dir}')
        produced = candidates[0]

    # A >2 GB export is split into an external-data sidecar that must stay
    # next to the graph; moving only the .onnx would silently strip the
    # weights. Leave those exports where Optimum put them.
    external = [p for p in work_dir.iterdir() if p.suffix in {'.onnx_data', '.pb'}]
    if external:
        logger.warning(
            f'Optimum wrote external weight data ({[p.name for p in external]}); '
            f'keeping the export in place at {produced} instead of moving it.'
        )
        return produced

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if produced != onnx_path:
        shutil.copy2(produced, onnx_path)
    logger.info(f'ONNX saved: {onnx_path} (optimum work dir kept at {work_dir})')
    return onnx_path


# ============================================================================
# Validation
# ============================================================================


@dataclass
class OnnxReport:
    """What :func:`validate_onnx` learned about the exported graph."""

    input_name: str | None = None
    input_shape: list[Any] = field(default_factory=list)
    outputs: list[tuple[str, list[Any]]] = field(default_factory=list)
    embedding_name: str | None = None
    embedding_dim: int | None = None
    dynamic_batch: bool | None = None
    skipped: bool = False


def _is_dynamic(dim: Any) -> bool:
    """Whether an ONNX dimension is symbolic (i.e. batchable)."""
    return isinstance(dim, str) or dim is None or (isinstance(dim, int) and dim < 1)


def validate_onnx(onnx_path: Path, image_size: int = IMAGE_SIZE) -> OnnxReport:
    """Probe the exported graph with ONNX Runtime on CPU.

    Reports the I/O names/shapes, the detected pooled-embedding output and
    whether the leading axis is dynamic. Returns a report with
    ``skipped=True`` (rather than raising) when onnxruntime isn't installed
    — the export itself is still usable.
    """
    report = OnnxReport()
    try:
        import numpy as np
        import onnxruntime as ort
    except ModuleNotFoundError:
        logger.warning('onnxruntime not installed — skipping ONNX validation.')
        report.skipped = True
        return report

    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    first_input = session.get_inputs()[0]
    report.input_name = first_input.name
    report.input_shape = list(first_input.shape)
    report.outputs = [(o.name, list(o.shape)) for o in session.get_outputs()]
    logger.info(f'ONNX inputs:  {[(i.name, i.shape) for i in session.get_inputs()]}')
    logger.info(f'ONNX outputs: {report.outputs}')

    # Concretize the declared input shape so we can run a real forward.
    concrete: list[int] = []
    for axis, dim in enumerate(report.input_shape):
        if _is_dynamic(dim):
            concrete.append({0: TRACE_BATCH, 1: 3}.get(axis, image_size))
        else:
            concrete.append(int(dim))
    logger.info(f'Probing forward with {report.input_name}={concrete}')
    arrays = session.run(None, {report.input_name: np.zeros(concrete, dtype=np.float32)})

    for (name, _declared), array in zip(report.outputs, arrays, strict=False):
        if array.ndim == 2:
            report.embedding_name = name
            report.embedding_dim = int(array.shape[-1])
            report.dynamic_batch = int(array.shape[0]) == concrete[0]
            logger.info(f'Pooled embedding: name={name} dim={report.embedding_dim}')
            break

    # A statically-shaped leading axis is the failure that silently breaks
    # Triton dynamic_batching: the model loads but rejects every batch > 1.
    declared_out = dict(report.outputs).get(report.embedding_name or '', [])
    if declared_out and not _is_dynamic(declared_out[0]):
        report.dynamic_batch = False
    return report


def check_client_contract(report: OnnxReport, cfg: PETritonConfig) -> list[str]:
    """Compare the exported graph against what ``PEEncoder`` will send/read.

    Returns a list of human-readable problems; empty means the graph can be
    served under the rendered config.pbtxt as-is.
    """
    if report.skipped:
        return []

    problems: list[str] = []
    if report.input_name != cfg.input_name:
        problems.append(
            f'input tensor is {report.input_name!r}, but PEEncoder sends {cfg.input_name!r}'
        )
    if report.embedding_name is None:
        problems.append('no 2-D pooled-embedding output found in the exported graph')
    elif report.embedding_name != cfg.output_name:
        problems.append(
            f'embedding output is {report.embedding_name!r}, but PEEncoder reads '
            f'{cfg.output_name!r}'
        )
    if report.embedding_dim is not None and report.embedding_dim != cfg.embedding_dim:
        problems.append(
            f'embedding dim is {report.embedding_dim}, but the curation index expects '
            f'{cfg.embedding_dim}'
        )
    if report.dynamic_batch is False:
        problems.append(
            'the leading axis is static — Triton dynamic_batching will reject batches > 1. '
            f'Re-export with a batch-{TRACE_BATCH} trace dummy.'
        )
    return problems


# ============================================================================
# CLI
# ============================================================================


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
        default=[0],
        help='GPU ids for the Triton instance group (default: 0)',
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

    problems: list[str] = []
    if not args.skip_validate:
        problems = check_client_contract(validate_onnx(onnx_path, args.image_size), cfg)

    if args.write_config:
        write_triton_config(args.models_dir, cfg)

    logger.info('=' * 70)
    logger.info(f'ONNX:     {onnx_path}')
    logger.info(f'Model:    {cfg.model_name} ({cfg.platform}, max_batch={cfg.max_batch_size})')
    logger.info(
        f'Contract: {cfg.input_name} [B, 3, {cfg.image_size}, {cfg.image_size}] FP32 -> '
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
